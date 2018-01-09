
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy import array
5: from numpy.testing import (assert_array_almost_equal, assert_array_equal,
6:                            assert_allclose,
7:                            assert_equal, assert_, assert_array_less)
8: from pytest import raises as assert_raises
9: 
10: from scipy._lib._numpy_compat import suppress_warnings
11: from scipy import signal, fftpack
12: 
13: 
14: window_funcs = [
15:     ('boxcar', ()),
16:     ('triang', ()),
17:     ('parzen', ()),
18:     ('bohman', ()),
19:     ('blackman', ()),
20:     ('nuttall', ()),
21:     ('blackmanharris', ()),
22:     ('flattop', ()),
23:     ('bartlett', ()),
24:     ('hanning', ()),
25:     ('barthann', ()),
26:     ('hamming', ()),
27:     ('kaiser', (1,)),
28:     ('gaussian', (0.5,)),
29:     ('general_gaussian', (1.5, 2)),
30:     ('chebwin', (1,)),
31:     ('slepian', (2,)),
32:     ('cosine', ()),
33:     ('hann', ()),
34:     ('exponential', ()),
35:     ('tukey', (0.5,)),
36:     ]
37: 
38: 
39: class TestBartHann(object):
40: 
41:     def test_basic(self):
42:         assert_allclose(signal.barthann(6, sym=True),
43:                         [0, 0.35857354213752, 0.8794264578624801,
44:                          0.8794264578624801, 0.3585735421375199, 0])
45:         assert_allclose(signal.barthann(7),
46:                         [0, 0.27, 0.73, 1.0, 0.73, 0.27, 0])
47:         assert_allclose(signal.barthann(6, False),
48:                         [0, 0.27, 0.73, 1.0, 0.73, 0.27])
49: 
50: 
51: class TestBartlett(object):
52: 
53:     def test_basic(self):
54:         assert_allclose(signal.bartlett(6), [0, 0.4, 0.8, 0.8, 0.4, 0])
55:         assert_allclose(signal.bartlett(7), [0, 1/3, 2/3, 1.0, 2/3, 1/3, 0])
56:         assert_allclose(signal.bartlett(6, False),
57:                         [0, 1/3, 2/3, 1.0, 2/3, 1/3])
58: 
59: 
60: class TestBlackman(object):
61: 
62:     def test_basic(self):
63:         assert_allclose(signal.blackman(6, sym=False),
64:                         [0, 0.13, 0.63, 1.0, 0.63, 0.13], atol=1e-14)
65:         assert_allclose(signal.blackman(7, sym=False),
66:                         [0, 0.09045342435412804, 0.4591829575459636,
67:                          0.9203636180999081, 0.9203636180999081,
68:                          0.4591829575459636, 0.09045342435412804], atol=1e-8)
69:         assert_allclose(signal.blackman(6),
70:                         [0, 0.2007701432625305, 0.8492298567374694,
71:                          0.8492298567374694, 0.2007701432625305, 0],
72:                         atol=1e-14)
73:         assert_allclose(signal.blackman(7, True),
74:                         [0, 0.13, 0.63, 1.0, 0.63, 0.13, 0], atol=1e-14)
75: 
76: 
77: class TestBlackmanHarris(object):
78: 
79:     def test_basic(self):
80:         assert_allclose(signal.blackmanharris(6, False),
81:                         [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645])
82:         assert_allclose(signal.blackmanharris(7, sym=False),
83:                         [6.0e-05, 0.03339172347815117, 0.332833504298565,
84:                          0.8893697722232837, 0.8893697722232838,
85:                          0.3328335042985652, 0.03339172347815122])
86:         assert_allclose(signal.blackmanharris(6),
87:                         [6.0e-05, 0.1030114893456638, 0.7938335106543362,
88:                          0.7938335106543364, 0.1030114893456638, 6.0e-05])
89:         assert_allclose(signal.blackmanharris(7, sym=True),
90:                         [6.0e-05, 0.055645, 0.520575, 1.0, 0.520575, 0.055645,
91:                          6.0e-05])
92: 
93: 
94: class TestBohman(object):
95: 
96:     def test_basic(self):
97:         assert_allclose(signal.bohman(6),
98:                         [0, 0.1791238937062839, 0.8343114522576858,
99:                          0.8343114522576858, 0.1791238937062838, 0])
100:         assert_allclose(signal.bohman(7, sym=True),
101:                         [0, 0.1089977810442293, 0.6089977810442293, 1.0,
102:                          0.6089977810442295, 0.1089977810442293, 0])
103:         assert_allclose(signal.bohman(6, False),
104:                         [0, 0.1089977810442293, 0.6089977810442293, 1.0,
105:                          0.6089977810442295, 0.1089977810442293])
106: 
107: 
108: class TestBoxcar(object):
109: 
110:     def test_basic(self):
111:         assert_allclose(signal.boxcar(6), [1, 1, 1, 1, 1, 1])
112:         assert_allclose(signal.boxcar(7), [1, 1, 1, 1, 1, 1, 1])
113:         assert_allclose(signal.boxcar(6, False), [1, 1, 1, 1, 1, 1])
114: 
115: 
116: cheb_odd_true = array([0.200938, 0.107729, 0.134941, 0.165348,
117:                        0.198891, 0.235450, 0.274846, 0.316836,
118:                        0.361119, 0.407338, 0.455079, 0.503883,
119:                        0.553248, 0.602637, 0.651489, 0.699227,
120:                        0.745266, 0.789028, 0.829947, 0.867485,
121:                        0.901138, 0.930448, 0.955010, 0.974482,
122:                        0.988591, 0.997138, 1.000000, 0.997138,
123:                        0.988591, 0.974482, 0.955010, 0.930448,
124:                        0.901138, 0.867485, 0.829947, 0.789028,
125:                        0.745266, 0.699227, 0.651489, 0.602637,
126:                        0.553248, 0.503883, 0.455079, 0.407338,
127:                        0.361119, 0.316836, 0.274846, 0.235450,
128:                        0.198891, 0.165348, 0.134941, 0.107729,
129:                        0.200938])
130: 
131: cheb_even_true = array([0.203894, 0.107279, 0.133904,
132:                         0.163608, 0.196338, 0.231986,
133:                         0.270385, 0.311313, 0.354493,
134:                         0.399594, 0.446233, 0.493983,
135:                         0.542378, 0.590916, 0.639071,
136:                         0.686302, 0.732055, 0.775783,
137:                         0.816944, 0.855021, 0.889525,
138:                         0.920006, 0.946060, 0.967339,
139:                         0.983557, 0.994494, 1.000000,
140:                         1.000000, 0.994494, 0.983557,
141:                         0.967339, 0.946060, 0.920006,
142:                         0.889525, 0.855021, 0.816944,
143:                         0.775783, 0.732055, 0.686302,
144:                         0.639071, 0.590916, 0.542378,
145:                         0.493983, 0.446233, 0.399594,
146:                         0.354493, 0.311313, 0.270385,
147:                         0.231986, 0.196338, 0.163608,
148:                         0.133904, 0.107279, 0.203894])
149: 
150: 
151: class TestChebWin(object):
152: 
153:     def test_basic(self):
154:         with suppress_warnings() as sup:
155:             sup.filter(UserWarning, "This window is not suitable")
156:             assert_allclose(signal.chebwin(6, 100),
157:                             [0.1046401879356917, 0.5075781475823447, 1.0, 1.0,
158:                              0.5075781475823447, 0.1046401879356917])
159:             assert_allclose(signal.chebwin(7, 100),
160:                             [0.05650405062850233, 0.316608530648474,
161:                              0.7601208123539079, 1.0, 0.7601208123539079,
162:                              0.316608530648474, 0.05650405062850233])
163:             assert_allclose(signal.chebwin(6, 10),
164:                             [1.0, 0.6071201674458373, 0.6808391469897297,
165:                              0.6808391469897297, 0.6071201674458373, 1.0])
166:             assert_allclose(signal.chebwin(7, 10),
167:                             [1.0, 0.5190521247588651, 0.5864059018130382,
168:                              0.6101519801307441, 0.5864059018130382,
169:                              0.5190521247588651, 1.0])
170:             assert_allclose(signal.chebwin(6, 10, False),
171:                             [1.0, 0.5190521247588651, 0.5864059018130382,
172:                              0.6101519801307441, 0.5864059018130382,
173:                              0.5190521247588651])
174: 
175:     def test_cheb_odd_high_attenuation(self):
176:         with suppress_warnings() as sup:
177:             sup.filter(UserWarning, "This window is not suitable")
178:             cheb_odd = signal.chebwin(53, at=-40)
179:         assert_array_almost_equal(cheb_odd, cheb_odd_true, decimal=4)
180: 
181:     def test_cheb_even_high_attenuation(self):
182:         with suppress_warnings() as sup:
183:             sup.filter(UserWarning, "This window is not suitable")
184:             cheb_even = signal.chebwin(54, at=40)
185:         assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)
186: 
187:     def test_cheb_odd_low_attenuation(self):
188:         cheb_odd_low_at_true = array([1.000000, 0.519052, 0.586405,
189:                                       0.610151, 0.586405, 0.519052,
190:                                       1.000000])
191:         with suppress_warnings() as sup:
192:             sup.filter(UserWarning, "This window is not suitable")
193:             cheb_odd = signal.chebwin(7, at=10)
194:         assert_array_almost_equal(cheb_odd, cheb_odd_low_at_true, decimal=4)
195: 
196:     def test_cheb_even_low_attenuation(self):
197:         cheb_even_low_at_true = array([1.000000, 0.451924, 0.51027,
198:                                        0.541338, 0.541338, 0.51027,
199:                                        0.451924, 1.000000])
200:         with suppress_warnings() as sup:
201:             sup.filter(UserWarning, "This window is not suitable")
202:             cheb_even = signal.chebwin(8, at=-10)
203:         assert_array_almost_equal(cheb_even, cheb_even_low_at_true, decimal=4)
204: 
205: 
206: exponential_data = {
207:     (4, None, 0.2, False):
208:         array([4.53999297624848542e-05,
209:                6.73794699908546700e-03, 1.00000000000000000e+00,
210:                6.73794699908546700e-03]),
211:     (4, None, 0.2, True): array([0.00055308437014783, 0.0820849986238988,
212:                                  0.0820849986238988, 0.00055308437014783]),
213:     (4, None, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
214:                                   0.36787944117144233]),
215:     (4, None, 1.0, True): array([0.22313016014842982, 0.60653065971263342,
216:                                  0.60653065971263342, 0.22313016014842982]),
217:     (4, 2, 0.2, False):
218:         array([4.53999297624848542e-05, 6.73794699908546700e-03,
219:                1.00000000000000000e+00, 6.73794699908546700e-03]),
220:     (4, 2, 0.2, True): None,
221:     (4, 2, 1.0, False): array([0.1353352832366127, 0.36787944117144233, 1.,
222:                                0.36787944117144233]),
223:     (4, 2, 1.0, True): None,
224:     (5, None, 0.2, True):
225:         array([4.53999297624848542e-05,
226:                6.73794699908546700e-03, 1.00000000000000000e+00,
227:                6.73794699908546700e-03, 4.53999297624848542e-05]),
228:     (5, None, 1.0, True): array([0.1353352832366127, 0.36787944117144233, 1.,
229:                                  0.36787944117144233, 0.1353352832366127]),
230:     (5, 2, 0.2, True): None,
231:     (5, 2, 1.0, True): None
232: }
233: 
234: 
235: def test_exponential():
236:     for k, v in exponential_data.items():
237:         if v is None:
238:             assert_raises(ValueError, signal.exponential, *k)
239:         else:
240:             win = signal.exponential(*k)
241:             assert_allclose(win, v, rtol=1e-14)
242: 
243: 
244: class TestFlatTop(object):
245: 
246:     def test_basic(self):
247:         assert_allclose(signal.flattop(6, sym=False),
248:                         [-0.000421051, -0.051263156, 0.19821053, 1.0,
249:                          0.19821053, -0.051263156])
250:         assert_allclose(signal.flattop(7, sym=False),
251:                         [-0.000421051, -0.03684078115492348,
252:                          0.01070371671615342, 0.7808739149387698,
253:                          0.7808739149387698, 0.01070371671615342,
254:                          -0.03684078115492348])
255:         assert_allclose(signal.flattop(6),
256:                         [-0.000421051, -0.0677142520762119, 0.6068721525762117,
257:                          0.6068721525762117, -0.0677142520762119,
258:                          -0.000421051])
259:         assert_allclose(signal.flattop(7, True),
260:                         [-0.000421051, -0.051263156, 0.19821053, 1.0,
261:                          0.19821053, -0.051263156, -0.000421051])
262: 
263: 
264: class TestGaussian(object):
265: 
266:     def test_basic(self):
267:         assert_allclose(signal.gaussian(6, 1.0),
268:                         [0.04393693362340742, 0.3246524673583497,
269:                          0.8824969025845955, 0.8824969025845955,
270:                          0.3246524673583497, 0.04393693362340742])
271:         assert_allclose(signal.gaussian(7, 1.2),
272:                         [0.04393693362340742, 0.2493522087772962,
273:                          0.7066482778577162, 1.0, 0.7066482778577162,
274:                          0.2493522087772962, 0.04393693362340742])
275:         assert_allclose(signal.gaussian(7, 3),
276:                         [0.6065306597126334, 0.8007374029168081,
277:                          0.9459594689067654, 1.0, 0.9459594689067654,
278:                          0.8007374029168081, 0.6065306597126334])
279:         assert_allclose(signal.gaussian(6, 3, False),
280:                         [0.6065306597126334, 0.8007374029168081,
281:                          0.9459594689067654, 1.0, 0.9459594689067654,
282:                          0.8007374029168081])
283: 
284: 
285: class TestHamming(object):
286: 
287:     def test_basic(self):
288:         assert_allclose(signal.hamming(6, False),
289:                         [0.08, 0.31, 0.77, 1.0, 0.77, 0.31])
290:         assert_allclose(signal.hamming(7, sym=False),
291:                         [0.08, 0.2531946911449826, 0.6423596296199047,
292:                          0.9544456792351128, 0.9544456792351128,
293:                          0.6423596296199047, 0.2531946911449826])
294:         assert_allclose(signal.hamming(6),
295:                         [0.08, 0.3978521825875242, 0.9121478174124757,
296:                          0.9121478174124757, 0.3978521825875242, 0.08])
297:         assert_allclose(signal.hamming(7, sym=True),
298:                         [0.08, 0.31, 0.77, 1.0, 0.77, 0.31, 0.08])
299: 
300: 
301: class TestHann(object):
302: 
303:     def test_basic(self):
304:         assert_allclose(signal.hann(6, sym=False),
305:                         [0, 0.25, 0.75, 1.0, 0.75, 0.25])
306:         assert_allclose(signal.hann(7, sym=False),
307:                         [0, 0.1882550990706332, 0.6112604669781572,
308:                          0.9504844339512095, 0.9504844339512095,
309:                          0.6112604669781572, 0.1882550990706332])
310:         assert_allclose(signal.hann(6, True),
311:                         [0, 0.3454915028125263, 0.9045084971874737,
312:                          0.9045084971874737, 0.3454915028125263, 0])
313:         assert_allclose(signal.hann(7),
314:                         [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0])
315: 
316: 
317: class TestKaiser(object):
318: 
319:     def test_basic(self):
320:         assert_allclose(signal.kaiser(6, 0.5),
321:                         [0.9403061933191572, 0.9782962393705389,
322:                          0.9975765035372042, 0.9975765035372042,
323:                          0.9782962393705389, 0.9403061933191572])
324:         assert_allclose(signal.kaiser(7, 0.5),
325:                         [0.9403061933191572, 0.9732402256999829,
326:                          0.9932754654413773, 1.0, 0.9932754654413773,
327:                          0.9732402256999829, 0.9403061933191572])
328:         assert_allclose(signal.kaiser(6, 2.7),
329:                         [0.2603047507678832, 0.6648106293528054,
330:                          0.9582099802511439, 0.9582099802511439,
331:                          0.6648106293528054, 0.2603047507678832])
332:         assert_allclose(signal.kaiser(7, 2.7),
333:                         [0.2603047507678832, 0.5985765418119844,
334:                          0.8868495172060835, 1.0, 0.8868495172060835,
335:                          0.5985765418119844, 0.2603047507678832])
336:         assert_allclose(signal.kaiser(6, 2.7, False),
337:                         [0.2603047507678832, 0.5985765418119844,
338:                          0.8868495172060835, 1.0, 0.8868495172060835,
339:                          0.5985765418119844])
340: 
341: 
342: class TestNuttall(object):
343: 
344:     def test_basic(self):
345:         assert_allclose(signal.nuttall(6, sym=False),
346:                         [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
347:                          0.0613345])
348:         assert_allclose(signal.nuttall(7, sym=False),
349:                         [0.0003628, 0.03777576895352025, 0.3427276199688195,
350:                          0.8918518610776603, 0.8918518610776603,
351:                          0.3427276199688196, 0.0377757689535203])
352:         assert_allclose(signal.nuttall(6),
353:                         [0.0003628, 0.1105152530498718, 0.7982580969501282,
354:                          0.7982580969501283, 0.1105152530498719, 0.0003628])
355:         assert_allclose(signal.nuttall(7, True),
356:                         [0.0003628, 0.0613345, 0.5292298, 1.0, 0.5292298,
357:                          0.0613345, 0.0003628])
358: 
359: 
360: class TestParzen(object):
361: 
362:     def test_basic(self):
363:         assert_allclose(signal.parzen(6),
364:                         [0.009259259259259254, 0.25, 0.8611111111111112,
365:                          0.8611111111111112, 0.25, 0.009259259259259254])
366:         assert_allclose(signal.parzen(7, sym=True),
367:                         [0.00583090379008747, 0.1574344023323616,
368:                          0.6501457725947521, 1.0, 0.6501457725947521,
369:                          0.1574344023323616, 0.00583090379008747])
370:         assert_allclose(signal.parzen(6, False),
371:                         [0.00583090379008747, 0.1574344023323616,
372:                          0.6501457725947521, 1.0, 0.6501457725947521,
373:                          0.1574344023323616])
374: 
375: 
376: class TestTriang(object):
377: 
378:     def test_basic(self):
379: 
380:         assert_allclose(signal.triang(6, True),
381:                         [1/6, 1/2, 5/6, 5/6, 1/2, 1/6])
382:         assert_allclose(signal.triang(7),
383:                         [1/4, 1/2, 3/4, 1, 3/4, 1/2, 1/4])
384:         assert_allclose(signal.triang(6, sym=False),
385:                         [1/4, 1/2, 3/4, 1, 3/4, 1/2])
386: 
387: 
388: tukey_data = {
389:     (4, 0.5, True): array([0.0, 1.0, 1.0, 0.0]),
390:     (4, 0.9, True): array([0.0, 0.84312081893436686,
391:                            0.84312081893436686, 0.0]),
392:     (4, 1.0, True): array([0.0, 0.75, 0.75, 0.0]),
393:     (4, 0.5, False): array([0.0, 1.0, 1.0, 1.0]),
394:     (4, 0.9, False): array([0.0, 0.58682408883346526,
395:                             1.0, 0.58682408883346526]),
396:     (4, 1.0, False): array([0.0, 0.5, 1.0, 0.5]),
397:     (5, 0.0, True): array([1.0, 1.0, 1.0, 1.0, 1.0]),
398:     (5, 0.8, True): array([0.0, 0.69134171618254492,
399:                            1.0, 0.69134171618254492, 0.0]),
400:     (5, 1.0, True): array([0.0, 0.5, 1.0, 0.5, 0.0]),
401: 
402:     (6, 0): [1, 1, 1, 1, 1, 1],
403:     (7, 0): [1, 1, 1, 1, 1, 1, 1],
404:     (6, .25): [0, 1, 1, 1, 1, 0],
405:     (7, .25): [0, 1, 1, 1, 1, 1, 0],
406:     (6,): [0, 0.9045084971874737, 1.0, 1.0, 0.9045084971874735, 0],
407:     (7,): [0, 0.75, 1.0, 1.0, 1.0, 0.75, 0],
408:     (6, .75): [0, 0.5522642316338269, 1.0, 1.0, 0.5522642316338267, 0],
409:     (7, .75): [0, 0.4131759111665348, 0.9698463103929542, 1.0,
410:                0.9698463103929542, 0.4131759111665347, 0],
411:     (6, 1): [0, 0.3454915028125263, 0.9045084971874737, 0.9045084971874737,
412:              0.3454915028125263, 0],
413:     (7, 1): [0, 0.25, 0.75, 1.0, 0.75, 0.25, 0],
414: }
415: 
416: 
417: class TestTukey(object):
418: 
419:     def test_basic(self):
420:         # Test against hardcoded data
421:         for k, v in tukey_data.items():
422:             if v is None:
423:                 assert_raises(ValueError, signal.tukey, *k)
424:             else:
425:                 win = signal.tukey(*k)
426:                 assert_allclose(win, v, rtol=1e-14)
427: 
428:     def test_extremes(self):
429:         # Test extremes of alpha correspond to boxcar and hann
430:         tuk0 = signal.tukey(100, 0)
431:         box0 = signal.boxcar(100)
432:         assert_array_almost_equal(tuk0, box0)
433: 
434:         tuk1 = signal.tukey(100, 1)
435:         han1 = signal.hann(100)
436:         assert_array_almost_equal(tuk1, han1)
437: 
438: 
439: class TestGetWindow(object):
440: 
441:     def test_boxcar(self):
442:         w = signal.get_window('boxcar', 12)
443:         assert_array_equal(w, np.ones_like(w))
444: 
445:         # window is a tuple of len 1
446:         w = signal.get_window(('boxcar',), 16)
447:         assert_array_equal(w, np.ones_like(w))
448: 
449:     def test_cheb_odd(self):
450:         with suppress_warnings() as sup:
451:             sup.filter(UserWarning, "This window is not suitable")
452:             w = signal.get_window(('chebwin', -40), 53, fftbins=False)
453:         assert_array_almost_equal(w, cheb_odd_true, decimal=4)
454: 
455:     def test_cheb_even(self):
456:         with suppress_warnings() as sup:
457:             sup.filter(UserWarning, "This window is not suitable")
458:             w = signal.get_window(('chebwin', 40), 54, fftbins=False)
459:         assert_array_almost_equal(w, cheb_even_true, decimal=4)
460: 
461:     def test_kaiser_float(self):
462:         win1 = signal.get_window(7.2, 64)
463:         win2 = signal.kaiser(64, 7.2, False)
464:         assert_allclose(win1, win2)
465: 
466:     def test_invalid_inputs(self):
467:         # Window is not a float, tuple, or string
468:         assert_raises(ValueError, signal.get_window, set('hann'), 8)
469: 
470:         # Unknown window type error
471:         assert_raises(ValueError, signal.get_window, 'broken', 4)
472: 
473:     def test_array_as_window(self):
474:         # github issue 3603
475:         osfactor = 128
476:         sig = np.arange(128)
477: 
478:         win = signal.get_window(('kaiser', 8.0), osfactor // 2)
479:         assert_raises(ValueError, signal.resample,
480:                       (sig, len(sig) * osfactor), {'window': win})
481: 
482: 
483: def test_windowfunc_basics():
484:     for window_name, params in window_funcs:
485:         window = getattr(signal, window_name)
486:         with suppress_warnings() as sup:
487:             sup.filter(UserWarning, "This window is not suitable")
488:             # Check symmetry for odd and even lengths
489:             w1 = window(8, *params, sym=True)
490:             w2 = window(7, *params, sym=False)
491:             assert_array_almost_equal(w1[:-1], w2)
492: 
493:             w1 = window(9, *params, sym=True)
494:             w2 = window(8, *params, sym=False)
495:             assert_array_almost_equal(w1[:-1], w2)
496: 
497:             # Check that functions run and output lengths are correct
498:             assert_equal(len(window(6, *params, sym=True)), 6)
499:             assert_equal(len(window(6, *params, sym=False)), 6)
500:             assert_equal(len(window(7, *params, sym=True)), 7)
501:             assert_equal(len(window(7, *params, sym=False)), 7)
502: 
503:             # Check invalid lengths
504:             assert_raises(ValueError, window, 5.5, *params)
505:             assert_raises(ValueError, window, -7, *params)
506: 
507:             # Check degenerate cases
508:             assert_array_equal(window(0, *params, sym=True), [])
509:             assert_array_equal(window(0, *params, sym=False), [])
510:             assert_array_equal(window(1, *params, sym=True), [1])
511:             assert_array_equal(window(1, *params, sym=False), [1])
512: 
513:             # Check dtype
514:             assert_(window(0, *params, sym=True).dtype == 'float')
515:             assert_(window(0, *params, sym=False).dtype == 'float')
516:             assert_(window(1, *params, sym=True).dtype == 'float')
517:             assert_(window(1, *params, sym=False).dtype == 'float')
518:             assert_(window(6, *params, sym=True).dtype == 'float')
519:             assert_(window(6, *params, sym=False).dtype == 'float')
520: 
521:             # Check normalization
522:             assert_array_less(window(10, *params, sym=True), 1.01)
523:             assert_array_less(window(10, *params, sym=False), 1.01)
524:             assert_array_less(window(9, *params, sym=True), 1.01)
525:             assert_array_less(window(9, *params, sym=False), 1.01)
526: 
527:             # Check that DFT-even spectrum is purely real for odd and even
528:             assert_allclose(fftpack.fft(window(10, *params, sym=False)).imag,
529:                             0, atol=1e-14)
530:             assert_allclose(fftpack.fft(window(11, *params, sym=False)).imag,
531:                             0, atol=1e-14)
532: 
533: 
534: def test_needs_params():
535:     for winstr in ['kaiser', 'ksr', 'gaussian', 'gauss', 'gss',
536:                    'general gaussian', 'general_gaussian',
537:                    'general gauss', 'general_gauss', 'ggs',
538:                    'slepian', 'optimal', 'slep', 'dss', 'dpss',
539:                    'chebwin', 'cheb', 'exponential', 'poisson', 'tukey',
540:                    'tuk']:
541:         assert_raises(ValueError, signal.get_window, winstr, 7)
542: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353845 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_353845) is not StypyTypeError):

    if (import_353845 != 'pyd_module'):
        __import__(import_353845)
        sys_modules_353846 = sys.modules[import_353845]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_353846.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_353845)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import array' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353847 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_353847) is not StypyTypeError):

    if (import_353847 != 'pyd_module'):
        __import__(import_353847)
        sys_modules_353848 = sys.modules[import_353847]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_353848.module_type_store, module_type_store, ['array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_353848, sys_modules_353848.module_type_store, module_type_store)
    else:
        from numpy import array

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['array'], [array])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_353847)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose, assert_equal, assert_, assert_array_less' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353849 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_353849) is not StypyTypeError):

    if (import_353849 != 'pyd_module'):
        __import__(import_353849)
        sys_modules_353850 = sys.modules[import_353849]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_353850.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_array_equal', 'assert_allclose', 'assert_equal', 'assert_', 'assert_array_less'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_353850, sys_modules_353850.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose, assert_equal, assert_, assert_array_less

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_array_equal', 'assert_allclose', 'assert_equal', 'assert_', 'assert_array_less'], [assert_array_almost_equal, assert_array_equal, assert_allclose, assert_equal, assert_, assert_array_less])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_353849)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353851 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_353851) is not StypyTypeError):

    if (import_353851 != 'pyd_module'):
        __import__(import_353851)
        sys_modules_353852 = sys.modules[import_353851]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_353852.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_353852, sys_modules_353852.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_353851)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat')

if (type(import_353853) is not StypyTypeError):

    if (import_353853 != 'pyd_module'):
        __import__(import_353853)
        sys_modules_353854 = sys.modules[import_353853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', sys_modules_353854.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_353854, sys_modules_353854.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', import_353853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy import signal, fftpack' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_353855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy')

if (type(import_353855) is not StypyTypeError):

    if (import_353855 != 'pyd_module'):
        __import__(import_353855)
        sys_modules_353856 = sys.modules[import_353855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', sys_modules_353856.module_type_store, module_type_store, ['signal', 'fftpack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_353856, sys_modules_353856.module_type_store, module_type_store)
    else:
        from scipy import signal, fftpack

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', None, module_type_store, ['signal', 'fftpack'], [signal, fftpack])

else:
    # Assigning a type to the variable 'scipy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', import_353855)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


# Assigning a List to a Name (line 14):

# Obtaining an instance of the builtin type 'list' (line 14)
list_353857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_353858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
str_353859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'str', 'boxcar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), tuple_353858, str_353859)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_353860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), tuple_353858, tuple_353860)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353858)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_353861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_353862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'str', 'triang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 5), tuple_353861, str_353862)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_353863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 5), tuple_353861, tuple_353863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353861)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_353864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_353865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 5), 'str', 'parzen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), tuple_353864, str_353865)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_353866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), tuple_353864, tuple_353866)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353864)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_353867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_353868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'str', 'bohman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), tuple_353867, str_353868)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_353869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), tuple_353867, tuple_353869)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353867)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_353870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_353871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 5), 'str', 'blackman')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 5), tuple_353870, str_353871)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_353872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 5), tuple_353870, tuple_353872)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353870)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_353873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
str_353874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 5), 'str', 'nuttall')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 5), tuple_353873, str_353874)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_353875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 5), tuple_353873, tuple_353875)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353873)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_353876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
str_353877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 5), 'str', 'blackmanharris')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), tuple_353876, str_353877)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_353878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 5), tuple_353876, tuple_353878)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353876)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_353879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_353880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 5), 'str', 'flattop')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), tuple_353879, str_353880)
# Adding element type (line 22)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_353881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 5), tuple_353879, tuple_353881)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353879)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_353882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_353883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'str', 'bartlett')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), tuple_353882, str_353883)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_353884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 5), tuple_353882, tuple_353884)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353882)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_353885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_353886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'str', 'hanning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), tuple_353885, str_353886)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_353887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 5), tuple_353885, tuple_353887)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353885)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_353888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
str_353889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'str', 'barthann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), tuple_353888, str_353889)
# Adding element type (line 25)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_353890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), tuple_353888, tuple_353890)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353888)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_353891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
str_353892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'str', 'hamming')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), tuple_353891, str_353892)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_353893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), tuple_353891, tuple_353893)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353891)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_353894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
str_353895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'str', 'kaiser')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), tuple_353894, str_353895)
# Adding element type (line 27)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_353896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
int_353897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_353896, int_353897)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 5), tuple_353894, tuple_353896)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353894)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_353898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
str_353899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 5), 'str', 'gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), tuple_353898, str_353899)
# Adding element type (line 28)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_353900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
float_353901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 18), tuple_353900, float_353901)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 5), tuple_353898, tuple_353900)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353898)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_353902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
str_353903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 5), 'str', 'general_gaussian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), tuple_353902, str_353903)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_353904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
float_353905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), tuple_353904, float_353905)
# Adding element type (line 29)
int_353906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), tuple_353904, int_353906)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 5), tuple_353902, tuple_353904)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353902)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_353907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_353908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 5), 'str', 'chebwin')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 5), tuple_353907, str_353908)
# Adding element type (line 30)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_353909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
int_353910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 17), tuple_353909, int_353910)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 5), tuple_353907, tuple_353909)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353907)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_353911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
str_353912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'str', 'slepian')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 5), tuple_353911, str_353912)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_353913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
int_353914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 17), tuple_353913, int_353914)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 5), tuple_353911, tuple_353913)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353911)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_353915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_353916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 5), 'str', 'cosine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 5), tuple_353915, str_353916)
# Adding element type (line 32)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_353917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 5), tuple_353915, tuple_353917)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353915)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_353918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_353919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 5), 'str', 'hann')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 5), tuple_353918, str_353919)
# Adding element type (line 33)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_353920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 5), tuple_353918, tuple_353920)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353918)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_353921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
str_353922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 5), 'str', 'exponential')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 5), tuple_353921, str_353922)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_353923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 5), tuple_353921, tuple_353923)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353921)
# Adding element type (line 14)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_353924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_353925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 5), 'str', 'tukey')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 5), tuple_353924, str_353925)
# Adding element type (line 35)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_353926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
float_353927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), tuple_353926, float_353927)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 5), tuple_353924, tuple_353926)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 15), list_353857, tuple_353924)

# Assigning a type to the variable 'window_funcs' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'window_funcs', list_353857)
# Declaration of the 'TestBartHann' class

class TestBartHann(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBartHann.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBartHann.test_basic')
        TestBartHann.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBartHann.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBartHann.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBartHann.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to barthann(...): (line 42)
        # Processing the call arguments (line 42)
        int_353931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 40), 'int')
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'True' (line 42)
        True_353932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 47), 'True', False)
        keyword_353933 = True_353932
        kwargs_353934 = {'sym': keyword_353933}
        # Getting the type of 'signal' (line 42)
        signal_353929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'signal', False)
        # Obtaining the member 'barthann' of a type (line 42)
        barthann_353930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), signal_353929, 'barthann')
        # Calling barthann(args, kwargs) (line 42)
        barthann_call_result_353935 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), barthann_353930, *[int_353931], **kwargs_353934)
        
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_353936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        int_353937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, int_353937)
        # Adding element type (line 43)
        float_353938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, float_353938)
        # Adding element type (line 43)
        float_353939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, float_353939)
        # Adding element type (line 43)
        float_353940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, float_353940)
        # Adding element type (line 43)
        float_353941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, float_353941)
        # Adding element type (line 43)
        int_353942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 24), list_353936, int_353942)
        
        # Processing the call keyword arguments (line 42)
        kwargs_353943 = {}
        # Getting the type of 'assert_allclose' (line 42)
        assert_allclose_353928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 42)
        assert_allclose_call_result_353944 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_allclose_353928, *[barthann_call_result_353935, list_353936], **kwargs_353943)
        
        
        # Call to assert_allclose(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to barthann(...): (line 45)
        # Processing the call arguments (line 45)
        int_353948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 40), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_353949 = {}
        # Getting the type of 'signal' (line 45)
        signal_353946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'signal', False)
        # Obtaining the member 'barthann' of a type (line 45)
        barthann_353947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), signal_353946, 'barthann')
        # Calling barthann(args, kwargs) (line 45)
        barthann_call_result_353950 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), barthann_353947, *[int_353948], **kwargs_353949)
        
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_353951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        # Adding element type (line 46)
        int_353952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, int_353952)
        # Adding element type (line 46)
        float_353953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, float_353953)
        # Adding element type (line 46)
        float_353954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, float_353954)
        # Adding element type (line 46)
        float_353955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, float_353955)
        # Adding element type (line 46)
        float_353956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, float_353956)
        # Adding element type (line 46)
        float_353957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, float_353957)
        # Adding element type (line 46)
        int_353958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 24), list_353951, int_353958)
        
        # Processing the call keyword arguments (line 45)
        kwargs_353959 = {}
        # Getting the type of 'assert_allclose' (line 45)
        assert_allclose_353945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 45)
        assert_allclose_call_result_353960 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_allclose_353945, *[barthann_call_result_353950, list_353951], **kwargs_353959)
        
        
        # Call to assert_allclose(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to barthann(...): (line 47)
        # Processing the call arguments (line 47)
        int_353964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
        # Getting the type of 'False' (line 47)
        False_353965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 43), 'False', False)
        # Processing the call keyword arguments (line 47)
        kwargs_353966 = {}
        # Getting the type of 'signal' (line 47)
        signal_353962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'signal', False)
        # Obtaining the member 'barthann' of a type (line 47)
        barthann_353963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), signal_353962, 'barthann')
        # Calling barthann(args, kwargs) (line 47)
        barthann_call_result_353967 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), barthann_353963, *[int_353964, False_353965], **kwargs_353966)
        
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_353968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        int_353969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, int_353969)
        # Adding element type (line 48)
        float_353970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, float_353970)
        # Adding element type (line 48)
        float_353971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, float_353971)
        # Adding element type (line 48)
        float_353972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, float_353972)
        # Adding element type (line 48)
        float_353973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, float_353973)
        # Adding element type (line 48)
        float_353974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 24), list_353968, float_353974)
        
        # Processing the call keyword arguments (line 47)
        kwargs_353975 = {}
        # Getting the type of 'assert_allclose' (line 47)
        assert_allclose_353961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 47)
        assert_allclose_call_result_353976 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), assert_allclose_353961, *[barthann_call_result_353967, list_353968], **kwargs_353975)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_353977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_353977


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 0, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBartHann.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBartHann' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'TestBartHann', TestBartHann)
# Declaration of the 'TestBartlett' class

class TestBartlett(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBartlett.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBartlett.test_basic')
        TestBartlett.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBartlett.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBartlett.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBartlett.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to bartlett(...): (line 54)
        # Processing the call arguments (line 54)
        int_353981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'int')
        # Processing the call keyword arguments (line 54)
        kwargs_353982 = {}
        # Getting the type of 'signal' (line 54)
        signal_353979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'signal', False)
        # Obtaining the member 'bartlett' of a type (line 54)
        bartlett_353980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), signal_353979, 'bartlett')
        # Calling bartlett(args, kwargs) (line 54)
        bartlett_call_result_353983 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), bartlett_353980, *[int_353981], **kwargs_353982)
        
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_353984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        int_353985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, int_353985)
        # Adding element type (line 54)
        float_353986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, float_353986)
        # Adding element type (line 54)
        float_353987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, float_353987)
        # Adding element type (line 54)
        float_353988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, float_353988)
        # Adding element type (line 54)
        float_353989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, float_353989)
        # Adding element type (line 54)
        int_353990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 44), list_353984, int_353990)
        
        # Processing the call keyword arguments (line 54)
        kwargs_353991 = {}
        # Getting the type of 'assert_allclose' (line 54)
        assert_allclose_353978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 54)
        assert_allclose_call_result_353992 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_allclose_353978, *[bartlett_call_result_353983, list_353984], **kwargs_353991)
        
        
        # Call to assert_allclose(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Call to bartlett(...): (line 55)
        # Processing the call arguments (line 55)
        int_353996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
        # Processing the call keyword arguments (line 55)
        kwargs_353997 = {}
        # Getting the type of 'signal' (line 55)
        signal_353994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'signal', False)
        # Obtaining the member 'bartlett' of a type (line 55)
        bartlett_353995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), signal_353994, 'bartlett')
        # Calling bartlett(args, kwargs) (line 55)
        bartlett_call_result_353998 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), bartlett_353995, *[int_353996], **kwargs_353997)
        
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_353999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        # Adding element type (line 55)
        int_354000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, int_354000)
        # Adding element type (line 55)
        int_354001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
        int_354002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 50), 'int')
        # Applying the binary operator 'div' (line 55)
        result_div_354003 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 48), 'div', int_354001, int_354002)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, result_div_354003)
        # Adding element type (line 55)
        int_354004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 53), 'int')
        int_354005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 55), 'int')
        # Applying the binary operator 'div' (line 55)
        result_div_354006 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 53), 'div', int_354004, int_354005)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, result_div_354006)
        # Adding element type (line 55)
        float_354007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, float_354007)
        # Adding element type (line 55)
        int_354008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 63), 'int')
        int_354009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 65), 'int')
        # Applying the binary operator 'div' (line 55)
        result_div_354010 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 63), 'div', int_354008, int_354009)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, result_div_354010)
        # Adding element type (line 55)
        int_354011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 68), 'int')
        int_354012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 70), 'int')
        # Applying the binary operator 'div' (line 55)
        result_div_354013 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 68), 'div', int_354011, int_354012)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, result_div_354013)
        # Adding element type (line 55)
        int_354014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 44), list_353999, int_354014)
        
        # Processing the call keyword arguments (line 55)
        kwargs_354015 = {}
        # Getting the type of 'assert_allclose' (line 55)
        assert_allclose_353993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 55)
        assert_allclose_call_result_354016 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_allclose_353993, *[bartlett_call_result_353998, list_353999], **kwargs_354015)
        
        
        # Call to assert_allclose(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to bartlett(...): (line 56)
        # Processing the call arguments (line 56)
        int_354020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 40), 'int')
        # Getting the type of 'False' (line 56)
        False_354021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 43), 'False', False)
        # Processing the call keyword arguments (line 56)
        kwargs_354022 = {}
        # Getting the type of 'signal' (line 56)
        signal_354018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'signal', False)
        # Obtaining the member 'bartlett' of a type (line 56)
        bartlett_354019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 24), signal_354018, 'bartlett')
        # Calling bartlett(args, kwargs) (line 56)
        bartlett_call_result_354023 = invoke(stypy.reporting.localization.Localization(__file__, 56, 24), bartlett_354019, *[int_354020, False_354021], **kwargs_354022)
        
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_354024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_354025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, int_354025)
        # Adding element type (line 57)
        int_354026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'int')
        int_354027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 30), 'int')
        # Applying the binary operator 'div' (line 57)
        result_div_354028 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 28), 'div', int_354026, int_354027)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, result_div_354028)
        # Adding element type (line 57)
        int_354029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'int')
        int_354030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 35), 'int')
        # Applying the binary operator 'div' (line 57)
        result_div_354031 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 33), 'div', int_354029, int_354030)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, result_div_354031)
        # Adding element type (line 57)
        float_354032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, float_354032)
        # Adding element type (line 57)
        int_354033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 43), 'int')
        int_354034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 45), 'int')
        # Applying the binary operator 'div' (line 57)
        result_div_354035 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 43), 'div', int_354033, int_354034)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, result_div_354035)
        # Adding element type (line 57)
        int_354036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 48), 'int')
        int_354037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 50), 'int')
        # Applying the binary operator 'div' (line 57)
        result_div_354038 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 48), 'div', int_354036, int_354037)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 24), list_354024, result_div_354038)
        
        # Processing the call keyword arguments (line 56)
        kwargs_354039 = {}
        # Getting the type of 'assert_allclose' (line 56)
        assert_allclose_354017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 56)
        assert_allclose_call_result_354040 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), assert_allclose_354017, *[bartlett_call_result_354023, list_354024], **kwargs_354039)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_354041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354041


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 51, 0, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBartlett.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBartlett' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'TestBartlett', TestBartlett)
# Declaration of the 'TestBlackman' class

class TestBlackman(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlackman.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBlackman.test_basic')
        TestBlackman.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlackman.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlackman.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlackman.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to blackman(...): (line 63)
        # Processing the call arguments (line 63)
        int_354045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 40), 'int')
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'False' (line 63)
        False_354046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'False', False)
        keyword_354047 = False_354046
        kwargs_354048 = {'sym': keyword_354047}
        # Getting the type of 'signal' (line 63)
        signal_354043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'signal', False)
        # Obtaining the member 'blackman' of a type (line 63)
        blackman_354044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), signal_354043, 'blackman')
        # Calling blackman(args, kwargs) (line 63)
        blackman_call_result_354049 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), blackman_354044, *[int_354045], **kwargs_354048)
        
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_354050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        int_354051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, int_354051)
        # Adding element type (line 64)
        float_354052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, float_354052)
        # Adding element type (line 64)
        float_354053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, float_354053)
        # Adding element type (line 64)
        float_354054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, float_354054)
        # Adding element type (line 64)
        float_354055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, float_354055)
        # Adding element type (line 64)
        float_354056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 24), list_354050, float_354056)
        
        # Processing the call keyword arguments (line 63)
        float_354057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 63), 'float')
        keyword_354058 = float_354057
        kwargs_354059 = {'atol': keyword_354058}
        # Getting the type of 'assert_allclose' (line 63)
        assert_allclose_354042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 63)
        assert_allclose_call_result_354060 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assert_allclose_354042, *[blackman_call_result_354049, list_354050], **kwargs_354059)
        
        
        # Call to assert_allclose(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to blackman(...): (line 65)
        # Processing the call arguments (line 65)
        int_354064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'int')
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'False' (line 65)
        False_354065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 47), 'False', False)
        keyword_354066 = False_354065
        kwargs_354067 = {'sym': keyword_354066}
        # Getting the type of 'signal' (line 65)
        signal_354062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'signal', False)
        # Obtaining the member 'blackman' of a type (line 65)
        blackman_354063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 24), signal_354062, 'blackman')
        # Calling blackman(args, kwargs) (line 65)
        blackman_call_result_354068 = invoke(stypy.reporting.localization.Localization(__file__, 65, 24), blackman_354063, *[int_354064], **kwargs_354067)
        
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_354069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        int_354070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, int_354070)
        # Adding element type (line 66)
        float_354071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354071)
        # Adding element type (line 66)
        float_354072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354072)
        # Adding element type (line 66)
        float_354073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354073)
        # Adding element type (line 66)
        float_354074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354074)
        # Adding element type (line 66)
        float_354075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354075)
        # Adding element type (line 66)
        float_354076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 24), list_354069, float_354076)
        
        # Processing the call keyword arguments (line 65)
        float_354077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 72), 'float')
        keyword_354078 = float_354077
        kwargs_354079 = {'atol': keyword_354078}
        # Getting the type of 'assert_allclose' (line 65)
        assert_allclose_354061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 65)
        assert_allclose_call_result_354080 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_allclose_354061, *[blackman_call_result_354068, list_354069], **kwargs_354079)
        
        
        # Call to assert_allclose(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Call to blackman(...): (line 69)
        # Processing the call arguments (line 69)
        int_354084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
        # Processing the call keyword arguments (line 69)
        kwargs_354085 = {}
        # Getting the type of 'signal' (line 69)
        signal_354082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'signal', False)
        # Obtaining the member 'blackman' of a type (line 69)
        blackman_354083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 24), signal_354082, 'blackman')
        # Calling blackman(args, kwargs) (line 69)
        blackman_call_result_354086 = invoke(stypy.reporting.localization.Localization(__file__, 69, 24), blackman_354083, *[int_354084], **kwargs_354085)
        
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_354087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        int_354088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, int_354088)
        # Adding element type (line 70)
        float_354089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, float_354089)
        # Adding element type (line 70)
        float_354090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, float_354090)
        # Adding element type (line 70)
        float_354091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, float_354091)
        # Adding element type (line 70)
        float_354092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, float_354092)
        # Adding element type (line 70)
        int_354093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 24), list_354087, int_354093)
        
        # Processing the call keyword arguments (line 69)
        float_354094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'float')
        keyword_354095 = float_354094
        kwargs_354096 = {'atol': keyword_354095}
        # Getting the type of 'assert_allclose' (line 69)
        assert_allclose_354081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 69)
        assert_allclose_call_result_354097 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), assert_allclose_354081, *[blackman_call_result_354086, list_354087], **kwargs_354096)
        
        
        # Call to assert_allclose(...): (line 73)
        # Processing the call arguments (line 73)
        
        # Call to blackman(...): (line 73)
        # Processing the call arguments (line 73)
        int_354101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 40), 'int')
        # Getting the type of 'True' (line 73)
        True_354102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 43), 'True', False)
        # Processing the call keyword arguments (line 73)
        kwargs_354103 = {}
        # Getting the type of 'signal' (line 73)
        signal_354099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'signal', False)
        # Obtaining the member 'blackman' of a type (line 73)
        blackman_354100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 24), signal_354099, 'blackman')
        # Calling blackman(args, kwargs) (line 73)
        blackman_call_result_354104 = invoke(stypy.reporting.localization.Localization(__file__, 73, 24), blackman_354100, *[int_354101, True_354102], **kwargs_354103)
        
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_354105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        int_354106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, int_354106)
        # Adding element type (line 74)
        float_354107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, float_354107)
        # Adding element type (line 74)
        float_354108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, float_354108)
        # Adding element type (line 74)
        float_354109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, float_354109)
        # Adding element type (line 74)
        float_354110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, float_354110)
        # Adding element type (line 74)
        float_354111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, float_354111)
        # Adding element type (line 74)
        int_354112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_354105, int_354112)
        
        # Processing the call keyword arguments (line 73)
        float_354113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 66), 'float')
        keyword_354114 = float_354113
        kwargs_354115 = {'atol': keyword_354114}
        # Getting the type of 'assert_allclose' (line 73)
        assert_allclose_354098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 73)
        assert_allclose_call_result_354116 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), assert_allclose_354098, *[blackman_call_result_354104, list_354105], **kwargs_354115)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_354117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354117


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 60, 0, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlackman.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBlackman' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'TestBlackman', TestBlackman)
# Declaration of the 'TestBlackmanHarris' class

class TestBlackmanHarris(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBlackmanHarris.test_basic')
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBlackmanHarris.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlackmanHarris.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Call to blackmanharris(...): (line 80)
        # Processing the call arguments (line 80)
        int_354121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 46), 'int')
        # Getting the type of 'False' (line 80)
        False_354122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'False', False)
        # Processing the call keyword arguments (line 80)
        kwargs_354123 = {}
        # Getting the type of 'signal' (line 80)
        signal_354119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'signal', False)
        # Obtaining the member 'blackmanharris' of a type (line 80)
        blackmanharris_354120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 24), signal_354119, 'blackmanharris')
        # Calling blackmanharris(args, kwargs) (line 80)
        blackmanharris_call_result_354124 = invoke(stypy.reporting.localization.Localization(__file__, 80, 24), blackmanharris_354120, *[int_354121, False_354122], **kwargs_354123)
        
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_354125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        float_354126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354126)
        # Adding element type (line 81)
        float_354127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354127)
        # Adding element type (line 81)
        float_354128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354128)
        # Adding element type (line 81)
        float_354129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354129)
        # Adding element type (line 81)
        float_354130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354130)
        # Adding element type (line 81)
        float_354131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 24), list_354125, float_354131)
        
        # Processing the call keyword arguments (line 80)
        kwargs_354132 = {}
        # Getting the type of 'assert_allclose' (line 80)
        assert_allclose_354118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 80)
        assert_allclose_call_result_354133 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_allclose_354118, *[blackmanharris_call_result_354124, list_354125], **kwargs_354132)
        
        
        # Call to assert_allclose(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to blackmanharris(...): (line 82)
        # Processing the call arguments (line 82)
        int_354137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 46), 'int')
        # Processing the call keyword arguments (line 82)
        # Getting the type of 'False' (line 82)
        False_354138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 53), 'False', False)
        keyword_354139 = False_354138
        kwargs_354140 = {'sym': keyword_354139}
        # Getting the type of 'signal' (line 82)
        signal_354135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'signal', False)
        # Obtaining the member 'blackmanharris' of a type (line 82)
        blackmanharris_354136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 24), signal_354135, 'blackmanharris')
        # Calling blackmanharris(args, kwargs) (line 82)
        blackmanharris_call_result_354141 = invoke(stypy.reporting.localization.Localization(__file__, 82, 24), blackmanharris_354136, *[int_354137], **kwargs_354140)
        
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_354142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        float_354143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354143)
        # Adding element type (line 83)
        float_354144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354144)
        # Adding element type (line 83)
        float_354145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354145)
        # Adding element type (line 83)
        float_354146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354146)
        # Adding element type (line 83)
        float_354147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354147)
        # Adding element type (line 83)
        float_354148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354148)
        # Adding element type (line 83)
        float_354149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 24), list_354142, float_354149)
        
        # Processing the call keyword arguments (line 82)
        kwargs_354150 = {}
        # Getting the type of 'assert_allclose' (line 82)
        assert_allclose_354134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 82)
        assert_allclose_call_result_354151 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert_allclose_354134, *[blackmanharris_call_result_354141, list_354142], **kwargs_354150)
        
        
        # Call to assert_allclose(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to blackmanharris(...): (line 86)
        # Processing the call arguments (line 86)
        int_354155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 46), 'int')
        # Processing the call keyword arguments (line 86)
        kwargs_354156 = {}
        # Getting the type of 'signal' (line 86)
        signal_354153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'signal', False)
        # Obtaining the member 'blackmanharris' of a type (line 86)
        blackmanharris_354154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), signal_354153, 'blackmanharris')
        # Calling blackmanharris(args, kwargs) (line 86)
        blackmanharris_call_result_354157 = invoke(stypy.reporting.localization.Localization(__file__, 86, 24), blackmanharris_354154, *[int_354155], **kwargs_354156)
        
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_354158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        float_354159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354159)
        # Adding element type (line 87)
        float_354160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354160)
        # Adding element type (line 87)
        float_354161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354161)
        # Adding element type (line 87)
        float_354162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354162)
        # Adding element type (line 87)
        float_354163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354163)
        # Adding element type (line 87)
        float_354164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 24), list_354158, float_354164)
        
        # Processing the call keyword arguments (line 86)
        kwargs_354165 = {}
        # Getting the type of 'assert_allclose' (line 86)
        assert_allclose_354152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 86)
        assert_allclose_call_result_354166 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assert_allclose_354152, *[blackmanharris_call_result_354157, list_354158], **kwargs_354165)
        
        
        # Call to assert_allclose(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to blackmanharris(...): (line 89)
        # Processing the call arguments (line 89)
        int_354170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 46), 'int')
        # Processing the call keyword arguments (line 89)
        # Getting the type of 'True' (line 89)
        True_354171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'True', False)
        keyword_354172 = True_354171
        kwargs_354173 = {'sym': keyword_354172}
        # Getting the type of 'signal' (line 89)
        signal_354168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'signal', False)
        # Obtaining the member 'blackmanharris' of a type (line 89)
        blackmanharris_354169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), signal_354168, 'blackmanharris')
        # Calling blackmanharris(args, kwargs) (line 89)
        blackmanharris_call_result_354174 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), blackmanharris_354169, *[int_354170], **kwargs_354173)
        
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_354175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_354176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354176)
        # Adding element type (line 90)
        float_354177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354177)
        # Adding element type (line 90)
        float_354178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354178)
        # Adding element type (line 90)
        float_354179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354179)
        # Adding element type (line 90)
        float_354180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354180)
        # Adding element type (line 90)
        float_354181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354181)
        # Adding element type (line 90)
        float_354182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_354175, float_354182)
        
        # Processing the call keyword arguments (line 89)
        kwargs_354183 = {}
        # Getting the type of 'assert_allclose' (line 89)
        assert_allclose_354167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 89)
        assert_allclose_call_result_354184 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), assert_allclose_354167, *[blackmanharris_call_result_354174, list_354175], **kwargs_354183)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_354185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354185)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354185


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 77, 0, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBlackmanHarris.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBlackmanHarris' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'TestBlackmanHarris', TestBlackmanHarris)
# Declaration of the 'TestBohman' class

class TestBohman(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBohman.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBohman.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBohman.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBohman.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBohman.test_basic')
        TestBohman.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBohman.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBohman.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBohman.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBohman.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBohman.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBohman.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBohman.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to bohman(...): (line 97)
        # Processing the call arguments (line 97)
        int_354189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 38), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_354190 = {}
        # Getting the type of 'signal' (line 97)
        signal_354187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'signal', False)
        # Obtaining the member 'bohman' of a type (line 97)
        bohman_354188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), signal_354187, 'bohman')
        # Calling bohman(args, kwargs) (line 97)
        bohman_call_result_354191 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), bohman_354188, *[int_354189], **kwargs_354190)
        
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_354192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        int_354193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, int_354193)
        # Adding element type (line 98)
        float_354194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, float_354194)
        # Adding element type (line 98)
        float_354195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, float_354195)
        # Adding element type (line 98)
        float_354196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, float_354196)
        # Adding element type (line 98)
        float_354197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, float_354197)
        # Adding element type (line 98)
        int_354198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 24), list_354192, int_354198)
        
        # Processing the call keyword arguments (line 97)
        kwargs_354199 = {}
        # Getting the type of 'assert_allclose' (line 97)
        assert_allclose_354186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 97)
        assert_allclose_call_result_354200 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_allclose_354186, *[bohman_call_result_354191, list_354192], **kwargs_354199)
        
        
        # Call to assert_allclose(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to bohman(...): (line 100)
        # Processing the call arguments (line 100)
        int_354204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 38), 'int')
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'True' (line 100)
        True_354205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'True', False)
        keyword_354206 = True_354205
        kwargs_354207 = {'sym': keyword_354206}
        # Getting the type of 'signal' (line 100)
        signal_354202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'signal', False)
        # Obtaining the member 'bohman' of a type (line 100)
        bohman_354203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 24), signal_354202, 'bohman')
        # Calling bohman(args, kwargs) (line 100)
        bohman_call_result_354208 = invoke(stypy.reporting.localization.Localization(__file__, 100, 24), bohman_354203, *[int_354204], **kwargs_354207)
        
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_354209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_354210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, int_354210)
        # Adding element type (line 101)
        float_354211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, float_354211)
        # Adding element type (line 101)
        float_354212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, float_354212)
        # Adding element type (line 101)
        float_354213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, float_354213)
        # Adding element type (line 101)
        float_354214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, float_354214)
        # Adding element type (line 101)
        float_354215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, float_354215)
        # Adding element type (line 101)
        int_354216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 24), list_354209, int_354216)
        
        # Processing the call keyword arguments (line 100)
        kwargs_354217 = {}
        # Getting the type of 'assert_allclose' (line 100)
        assert_allclose_354201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 100)
        assert_allclose_call_result_354218 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), assert_allclose_354201, *[bohman_call_result_354208, list_354209], **kwargs_354217)
        
        
        # Call to assert_allclose(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to bohman(...): (line 103)
        # Processing the call arguments (line 103)
        int_354222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 38), 'int')
        # Getting the type of 'False' (line 103)
        False_354223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'False', False)
        # Processing the call keyword arguments (line 103)
        kwargs_354224 = {}
        # Getting the type of 'signal' (line 103)
        signal_354220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'signal', False)
        # Obtaining the member 'bohman' of a type (line 103)
        bohman_354221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 24), signal_354220, 'bohman')
        # Calling bohman(args, kwargs) (line 103)
        bohman_call_result_354225 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), bohman_354221, *[int_354222, False_354223], **kwargs_354224)
        
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_354226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        int_354227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, int_354227)
        # Adding element type (line 104)
        float_354228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, float_354228)
        # Adding element type (line 104)
        float_354229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, float_354229)
        # Adding element type (line 104)
        float_354230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, float_354230)
        # Adding element type (line 104)
        float_354231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, float_354231)
        # Adding element type (line 104)
        float_354232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 24), list_354226, float_354232)
        
        # Processing the call keyword arguments (line 103)
        kwargs_354233 = {}
        # Getting the type of 'assert_allclose' (line 103)
        assert_allclose_354219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 103)
        assert_allclose_call_result_354234 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assert_allclose_354219, *[bohman_call_result_354225, list_354226], **kwargs_354233)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_354235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354235


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 94, 0, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBohman.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBohman' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'TestBohman', TestBohman)
# Declaration of the 'TestBoxcar' class

class TestBoxcar(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_function_name', 'TestBoxcar.test_basic')
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBoxcar.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBoxcar.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to boxcar(...): (line 111)
        # Processing the call arguments (line 111)
        int_354239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
        # Processing the call keyword arguments (line 111)
        kwargs_354240 = {}
        # Getting the type of 'signal' (line 111)
        signal_354237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'signal', False)
        # Obtaining the member 'boxcar' of a type (line 111)
        boxcar_354238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), signal_354237, 'boxcar')
        # Calling boxcar(args, kwargs) (line 111)
        boxcar_call_result_354241 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), boxcar_354238, *[int_354239], **kwargs_354240)
        
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_354242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        int_354243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354243)
        # Adding element type (line 111)
        int_354244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354244)
        # Adding element type (line 111)
        int_354245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354245)
        # Adding element type (line 111)
        int_354246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354246)
        # Adding element type (line 111)
        int_354247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354247)
        # Adding element type (line 111)
        int_354248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 42), list_354242, int_354248)
        
        # Processing the call keyword arguments (line 111)
        kwargs_354249 = {}
        # Getting the type of 'assert_allclose' (line 111)
        assert_allclose_354236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 111)
        assert_allclose_call_result_354250 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assert_allclose_354236, *[boxcar_call_result_354241, list_354242], **kwargs_354249)
        
        
        # Call to assert_allclose(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to boxcar(...): (line 112)
        # Processing the call arguments (line 112)
        int_354254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 38), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_354255 = {}
        # Getting the type of 'signal' (line 112)
        signal_354252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'signal', False)
        # Obtaining the member 'boxcar' of a type (line 112)
        boxcar_354253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 24), signal_354252, 'boxcar')
        # Calling boxcar(args, kwargs) (line 112)
        boxcar_call_result_354256 = invoke(stypy.reporting.localization.Localization(__file__, 112, 24), boxcar_354253, *[int_354254], **kwargs_354255)
        
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_354257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        int_354258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354258)
        # Adding element type (line 112)
        int_354259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354259)
        # Adding element type (line 112)
        int_354260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354260)
        # Adding element type (line 112)
        int_354261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354261)
        # Adding element type (line 112)
        int_354262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354262)
        # Adding element type (line 112)
        int_354263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354263)
        # Adding element type (line 112)
        int_354264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_354257, int_354264)
        
        # Processing the call keyword arguments (line 112)
        kwargs_354265 = {}
        # Getting the type of 'assert_allclose' (line 112)
        assert_allclose_354251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 112)
        assert_allclose_call_result_354266 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assert_allclose_354251, *[boxcar_call_result_354256, list_354257], **kwargs_354265)
        
        
        # Call to assert_allclose(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to boxcar(...): (line 113)
        # Processing the call arguments (line 113)
        int_354270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 38), 'int')
        # Getting the type of 'False' (line 113)
        False_354271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 41), 'False', False)
        # Processing the call keyword arguments (line 113)
        kwargs_354272 = {}
        # Getting the type of 'signal' (line 113)
        signal_354268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'signal', False)
        # Obtaining the member 'boxcar' of a type (line 113)
        boxcar_354269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), signal_354268, 'boxcar')
        # Calling boxcar(args, kwargs) (line 113)
        boxcar_call_result_354273 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), boxcar_354269, *[int_354270, False_354271], **kwargs_354272)
        
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_354274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        int_354275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354275)
        # Adding element type (line 113)
        int_354276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354276)
        # Adding element type (line 113)
        int_354277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354277)
        # Adding element type (line 113)
        int_354278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354278)
        # Adding element type (line 113)
        int_354279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354279)
        # Adding element type (line 113)
        int_354280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 49), list_354274, int_354280)
        
        # Processing the call keyword arguments (line 113)
        kwargs_354281 = {}
        # Getting the type of 'assert_allclose' (line 113)
        assert_allclose_354267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 113)
        assert_allclose_call_result_354282 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_allclose_354267, *[boxcar_call_result_354273, list_354274], **kwargs_354281)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_354283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354283


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 0, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBoxcar.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBoxcar' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'TestBoxcar', TestBoxcar)

# Assigning a Call to a Name (line 116):

# Call to array(...): (line 116)
# Processing the call arguments (line 116)

# Obtaining an instance of the builtin type 'list' (line 116)
list_354285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 116)
# Adding element type (line 116)
float_354286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354286)
# Adding element type (line 116)
float_354287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354287)
# Adding element type (line 116)
float_354288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354288)
# Adding element type (line 116)
float_354289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354289)
# Adding element type (line 116)
float_354290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354290)
# Adding element type (line 116)
float_354291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354291)
# Adding element type (line 116)
float_354292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354292)
# Adding element type (line 116)
float_354293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354293)
# Adding element type (line 116)
float_354294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354294)
# Adding element type (line 116)
float_354295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354295)
# Adding element type (line 116)
float_354296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354296)
# Adding element type (line 116)
float_354297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354297)
# Adding element type (line 116)
float_354298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354298)
# Adding element type (line 116)
float_354299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354299)
# Adding element type (line 116)
float_354300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354300)
# Adding element type (line 116)
float_354301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354301)
# Adding element type (line 116)
float_354302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354302)
# Adding element type (line 116)
float_354303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354303)
# Adding element type (line 116)
float_354304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354304)
# Adding element type (line 116)
float_354305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354305)
# Adding element type (line 116)
float_354306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354306)
# Adding element type (line 116)
float_354307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354307)
# Adding element type (line 116)
float_354308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354308)
# Adding element type (line 116)
float_354309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354309)
# Adding element type (line 116)
float_354310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354310)
# Adding element type (line 116)
float_354311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354311)
# Adding element type (line 116)
float_354312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354312)
# Adding element type (line 116)
float_354313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354313)
# Adding element type (line 116)
float_354314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354314)
# Adding element type (line 116)
float_354315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354315)
# Adding element type (line 116)
float_354316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354316)
# Adding element type (line 116)
float_354317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354317)
# Adding element type (line 116)
float_354318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354318)
# Adding element type (line 116)
float_354319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354319)
# Adding element type (line 116)
float_354320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354320)
# Adding element type (line 116)
float_354321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354321)
# Adding element type (line 116)
float_354322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354322)
# Adding element type (line 116)
float_354323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354323)
# Adding element type (line 116)
float_354324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354324)
# Adding element type (line 116)
float_354325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354325)
# Adding element type (line 116)
float_354326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354326)
# Adding element type (line 116)
float_354327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354327)
# Adding element type (line 116)
float_354328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354328)
# Adding element type (line 116)
float_354329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354329)
# Adding element type (line 116)
float_354330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354330)
# Adding element type (line 116)
float_354331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354331)
# Adding element type (line 116)
float_354332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354332)
# Adding element type (line 116)
float_354333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354333)
# Adding element type (line 116)
float_354334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354334)
# Adding element type (line 116)
float_354335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354335)
# Adding element type (line 116)
float_354336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354336)
# Adding element type (line 116)
float_354337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354337)
# Adding element type (line 116)
float_354338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 22), list_354285, float_354338)

# Processing the call keyword arguments (line 116)
kwargs_354339 = {}
# Getting the type of 'array' (line 116)
array_354284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'array', False)
# Calling array(args, kwargs) (line 116)
array_call_result_354340 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), array_354284, *[list_354285], **kwargs_354339)

# Assigning a type to the variable 'cheb_odd_true' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'cheb_odd_true', array_call_result_354340)

# Assigning a Call to a Name (line 131):

# Call to array(...): (line 131)
# Processing the call arguments (line 131)

# Obtaining an instance of the builtin type 'list' (line 131)
list_354342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 131)
# Adding element type (line 131)
float_354343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354343)
# Adding element type (line 131)
float_354344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354344)
# Adding element type (line 131)
float_354345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354345)
# Adding element type (line 131)
float_354346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354346)
# Adding element type (line 131)
float_354347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354347)
# Adding element type (line 131)
float_354348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354348)
# Adding element type (line 131)
float_354349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354349)
# Adding element type (line 131)
float_354350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354350)
# Adding element type (line 131)
float_354351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354351)
# Adding element type (line 131)
float_354352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354352)
# Adding element type (line 131)
float_354353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354353)
# Adding element type (line 131)
float_354354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354354)
# Adding element type (line 131)
float_354355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354355)
# Adding element type (line 131)
float_354356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354356)
# Adding element type (line 131)
float_354357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354357)
# Adding element type (line 131)
float_354358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354358)
# Adding element type (line 131)
float_354359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354359)
# Adding element type (line 131)
float_354360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354360)
# Adding element type (line 131)
float_354361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354361)
# Adding element type (line 131)
float_354362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354362)
# Adding element type (line 131)
float_354363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354363)
# Adding element type (line 131)
float_354364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354364)
# Adding element type (line 131)
float_354365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354365)
# Adding element type (line 131)
float_354366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354366)
# Adding element type (line 131)
float_354367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354367)
# Adding element type (line 131)
float_354368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354368)
# Adding element type (line 131)
float_354369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354369)
# Adding element type (line 131)
float_354370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354370)
# Adding element type (line 131)
float_354371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354371)
# Adding element type (line 131)
float_354372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354372)
# Adding element type (line 131)
float_354373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354373)
# Adding element type (line 131)
float_354374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354374)
# Adding element type (line 131)
float_354375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354375)
# Adding element type (line 131)
float_354376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354376)
# Adding element type (line 131)
float_354377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354377)
# Adding element type (line 131)
float_354378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354378)
# Adding element type (line 131)
float_354379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354379)
# Adding element type (line 131)
float_354380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354380)
# Adding element type (line 131)
float_354381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354381)
# Adding element type (line 131)
float_354382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354382)
# Adding element type (line 131)
float_354383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354383)
# Adding element type (line 131)
float_354384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354384)
# Adding element type (line 131)
float_354385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354385)
# Adding element type (line 131)
float_354386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354386)
# Adding element type (line 131)
float_354387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354387)
# Adding element type (line 131)
float_354388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354388)
# Adding element type (line 131)
float_354389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354389)
# Adding element type (line 131)
float_354390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354390)
# Adding element type (line 131)
float_354391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354391)
# Adding element type (line 131)
float_354392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354392)
# Adding element type (line 131)
float_354393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354393)
# Adding element type (line 131)
float_354394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354394)
# Adding element type (line 131)
float_354395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354395)
# Adding element type (line 131)
float_354396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 23), list_354342, float_354396)

# Processing the call keyword arguments (line 131)
kwargs_354397 = {}
# Getting the type of 'array' (line 131)
array_354341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'array', False)
# Calling array(args, kwargs) (line 131)
array_call_result_354398 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), array_354341, *[list_354342], **kwargs_354397)

# Assigning a type to the variable 'cheb_even_true' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'cheb_even_true', array_call_result_354398)
# Declaration of the 'TestChebWin' class

class TestChebWin(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 153, 4, False)
        # Assigning a type to the variable 'self' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChebWin.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_function_name', 'TestChebWin.test_basic')
        TestChebWin.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestChebWin.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChebWin.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to suppress_warnings(...): (line 154)
        # Processing the call keyword arguments (line 154)
        kwargs_354400 = {}
        # Getting the type of 'suppress_warnings' (line 154)
        suppress_warnings_354399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 154)
        suppress_warnings_call_result_354401 = invoke(stypy.reporting.localization.Localization(__file__, 154, 13), suppress_warnings_354399, *[], **kwargs_354400)
        
        with_354402 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 154, 13), suppress_warnings_call_result_354401, 'with parameter', '__enter__', '__exit__')

        if with_354402:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 154)
            enter___354403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 13), suppress_warnings_call_result_354401, '__enter__')
            with_enter_354404 = invoke(stypy.reporting.localization.Localization(__file__, 154, 13), enter___354403)
            # Assigning a type to the variable 'sup' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'sup', with_enter_354404)
            
            # Call to filter(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'UserWarning' (line 155)
            UserWarning_354407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 23), 'UserWarning', False)
            str_354408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 155)
            kwargs_354409 = {}
            # Getting the type of 'sup' (line 155)
            sup_354405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 155)
            filter_354406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), sup_354405, 'filter')
            # Calling filter(args, kwargs) (line 155)
            filter_call_result_354410 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), filter_354406, *[UserWarning_354407, str_354408], **kwargs_354409)
            
            
            # Call to assert_allclose(...): (line 156)
            # Processing the call arguments (line 156)
            
            # Call to chebwin(...): (line 156)
            # Processing the call arguments (line 156)
            int_354414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 43), 'int')
            int_354415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 46), 'int')
            # Processing the call keyword arguments (line 156)
            kwargs_354416 = {}
            # Getting the type of 'signal' (line 156)
            signal_354412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 156)
            chebwin_354413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 28), signal_354412, 'chebwin')
            # Calling chebwin(args, kwargs) (line 156)
            chebwin_call_result_354417 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), chebwin_354413, *[int_354414, int_354415], **kwargs_354416)
            
            
            # Obtaining an instance of the builtin type 'list' (line 157)
            list_354418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 157)
            # Adding element type (line 157)
            float_354419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354419)
            # Adding element type (line 157)
            float_354420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354420)
            # Adding element type (line 157)
            float_354421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 69), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354421)
            # Adding element type (line 157)
            float_354422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 74), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354422)
            # Adding element type (line 157)
            float_354423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354423)
            # Adding element type (line 157)
            float_354424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 28), list_354418, float_354424)
            
            # Processing the call keyword arguments (line 156)
            kwargs_354425 = {}
            # Getting the type of 'assert_allclose' (line 156)
            assert_allclose_354411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 156)
            assert_allclose_call_result_354426 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), assert_allclose_354411, *[chebwin_call_result_354417, list_354418], **kwargs_354425)
            
            
            # Call to assert_allclose(...): (line 159)
            # Processing the call arguments (line 159)
            
            # Call to chebwin(...): (line 159)
            # Processing the call arguments (line 159)
            int_354430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'int')
            int_354431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 46), 'int')
            # Processing the call keyword arguments (line 159)
            kwargs_354432 = {}
            # Getting the type of 'signal' (line 159)
            signal_354428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 159)
            chebwin_354429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 28), signal_354428, 'chebwin')
            # Calling chebwin(args, kwargs) (line 159)
            chebwin_call_result_354433 = invoke(stypy.reporting.localization.Localization(__file__, 159, 28), chebwin_354429, *[int_354430, int_354431], **kwargs_354432)
            
            
            # Obtaining an instance of the builtin type 'list' (line 160)
            list_354434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 160)
            # Adding element type (line 160)
            float_354435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354435)
            # Adding element type (line 160)
            float_354436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 50), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354436)
            # Adding element type (line 160)
            float_354437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354437)
            # Adding element type (line 160)
            float_354438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354438)
            # Adding element type (line 160)
            float_354439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354439)
            # Adding element type (line 160)
            float_354440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354440)
            # Adding element type (line 160)
            float_354441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 48), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 28), list_354434, float_354441)
            
            # Processing the call keyword arguments (line 159)
            kwargs_354442 = {}
            # Getting the type of 'assert_allclose' (line 159)
            assert_allclose_354427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 159)
            assert_allclose_call_result_354443 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), assert_allclose_354427, *[chebwin_call_result_354433, list_354434], **kwargs_354442)
            
            
            # Call to assert_allclose(...): (line 163)
            # Processing the call arguments (line 163)
            
            # Call to chebwin(...): (line 163)
            # Processing the call arguments (line 163)
            int_354447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 43), 'int')
            int_354448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 46), 'int')
            # Processing the call keyword arguments (line 163)
            kwargs_354449 = {}
            # Getting the type of 'signal' (line 163)
            signal_354445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 163)
            chebwin_354446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 28), signal_354445, 'chebwin')
            # Calling chebwin(args, kwargs) (line 163)
            chebwin_call_result_354450 = invoke(stypy.reporting.localization.Localization(__file__, 163, 28), chebwin_354446, *[int_354447, int_354448], **kwargs_354449)
            
            
            # Obtaining an instance of the builtin type 'list' (line 164)
            list_354451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 164)
            # Adding element type (line 164)
            float_354452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354452)
            # Adding element type (line 164)
            float_354453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 34), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354453)
            # Adding element type (line 164)
            float_354454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354454)
            # Adding element type (line 164)
            float_354455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354455)
            # Adding element type (line 164)
            float_354456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354456)
            # Adding element type (line 164)
            float_354457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 69), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), list_354451, float_354457)
            
            # Processing the call keyword arguments (line 163)
            kwargs_354458 = {}
            # Getting the type of 'assert_allclose' (line 163)
            assert_allclose_354444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 163)
            assert_allclose_call_result_354459 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), assert_allclose_354444, *[chebwin_call_result_354450, list_354451], **kwargs_354458)
            
            
            # Call to assert_allclose(...): (line 166)
            # Processing the call arguments (line 166)
            
            # Call to chebwin(...): (line 166)
            # Processing the call arguments (line 166)
            int_354463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 43), 'int')
            int_354464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 46), 'int')
            # Processing the call keyword arguments (line 166)
            kwargs_354465 = {}
            # Getting the type of 'signal' (line 166)
            signal_354461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 166)
            chebwin_354462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 28), signal_354461, 'chebwin')
            # Calling chebwin(args, kwargs) (line 166)
            chebwin_call_result_354466 = invoke(stypy.reporting.localization.Localization(__file__, 166, 28), chebwin_354462, *[int_354463, int_354464], **kwargs_354465)
            
            
            # Obtaining an instance of the builtin type 'list' (line 167)
            list_354467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 167)
            # Adding element type (line 167)
            float_354468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354468)
            # Adding element type (line 167)
            float_354469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 34), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354469)
            # Adding element type (line 167)
            float_354470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354470)
            # Adding element type (line 167)
            float_354471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354471)
            # Adding element type (line 167)
            float_354472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354472)
            # Adding element type (line 167)
            float_354473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354473)
            # Adding element type (line 167)
            float_354474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 28), list_354467, float_354474)
            
            # Processing the call keyword arguments (line 166)
            kwargs_354475 = {}
            # Getting the type of 'assert_allclose' (line 166)
            assert_allclose_354460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 166)
            assert_allclose_call_result_354476 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), assert_allclose_354460, *[chebwin_call_result_354466, list_354467], **kwargs_354475)
            
            
            # Call to assert_allclose(...): (line 170)
            # Processing the call arguments (line 170)
            
            # Call to chebwin(...): (line 170)
            # Processing the call arguments (line 170)
            int_354480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 43), 'int')
            int_354481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 46), 'int')
            # Getting the type of 'False' (line 170)
            False_354482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 50), 'False', False)
            # Processing the call keyword arguments (line 170)
            kwargs_354483 = {}
            # Getting the type of 'signal' (line 170)
            signal_354478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 170)
            chebwin_354479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 28), signal_354478, 'chebwin')
            # Calling chebwin(args, kwargs) (line 170)
            chebwin_call_result_354484 = invoke(stypy.reporting.localization.Localization(__file__, 170, 28), chebwin_354479, *[int_354480, int_354481, False_354482], **kwargs_354483)
            
            
            # Obtaining an instance of the builtin type 'list' (line 171)
            list_354485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 171)
            # Adding element type (line 171)
            float_354486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354486)
            # Adding element type (line 171)
            float_354487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354487)
            # Adding element type (line 171)
            float_354488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 54), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354488)
            # Adding element type (line 171)
            float_354489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354489)
            # Adding element type (line 171)
            float_354490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354490)
            # Adding element type (line 171)
            float_354491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 29), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 28), list_354485, float_354491)
            
            # Processing the call keyword arguments (line 170)
            kwargs_354492 = {}
            # Getting the type of 'assert_allclose' (line 170)
            assert_allclose_354477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 170)
            assert_allclose_call_result_354493 = invoke(stypy.reporting.localization.Localization(__file__, 170, 12), assert_allclose_354477, *[chebwin_call_result_354484, list_354485], **kwargs_354492)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 154)
            exit___354494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 13), suppress_warnings_call_result_354401, '__exit__')
            with_exit_354495 = invoke(stypy.reporting.localization.Localization(__file__, 154, 13), exit___354494, None, None, None)

        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 153)
        stypy_return_type_354496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354496


    @norecursion
    def test_cheb_odd_high_attenuation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_odd_high_attenuation'
        module_type_store = module_type_store.open_function_context('test_cheb_odd_high_attenuation', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_localization', localization)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_function_name', 'TestChebWin.test_cheb_odd_high_attenuation')
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_param_names_list', [])
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChebWin.test_cheb_odd_high_attenuation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.test_cheb_odd_high_attenuation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_odd_high_attenuation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_odd_high_attenuation(...)' code ##################

        
        # Call to suppress_warnings(...): (line 176)
        # Processing the call keyword arguments (line 176)
        kwargs_354498 = {}
        # Getting the type of 'suppress_warnings' (line 176)
        suppress_warnings_354497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 176)
        suppress_warnings_call_result_354499 = invoke(stypy.reporting.localization.Localization(__file__, 176, 13), suppress_warnings_354497, *[], **kwargs_354498)
        
        with_354500 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 176, 13), suppress_warnings_call_result_354499, 'with parameter', '__enter__', '__exit__')

        if with_354500:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 176)
            enter___354501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 13), suppress_warnings_call_result_354499, '__enter__')
            with_enter_354502 = invoke(stypy.reporting.localization.Localization(__file__, 176, 13), enter___354501)
            # Assigning a type to the variable 'sup' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'sup', with_enter_354502)
            
            # Call to filter(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'UserWarning' (line 177)
            UserWarning_354505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'UserWarning', False)
            str_354506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 177)
            kwargs_354507 = {}
            # Getting the type of 'sup' (line 177)
            sup_354503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 177)
            filter_354504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), sup_354503, 'filter')
            # Calling filter(args, kwargs) (line 177)
            filter_call_result_354508 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), filter_354504, *[UserWarning_354505, str_354506], **kwargs_354507)
            
            
            # Assigning a Call to a Name (line 178):
            
            # Call to chebwin(...): (line 178)
            # Processing the call arguments (line 178)
            int_354511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 38), 'int')
            # Processing the call keyword arguments (line 178)
            int_354512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'int')
            keyword_354513 = int_354512
            kwargs_354514 = {'at': keyword_354513}
            # Getting the type of 'signal' (line 178)
            signal_354509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 178)
            chebwin_354510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 23), signal_354509, 'chebwin')
            # Calling chebwin(args, kwargs) (line 178)
            chebwin_call_result_354515 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), chebwin_354510, *[int_354511], **kwargs_354514)
            
            # Assigning a type to the variable 'cheb_odd' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'cheb_odd', chebwin_call_result_354515)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 176)
            exit___354516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 13), suppress_warnings_call_result_354499, '__exit__')
            with_exit_354517 = invoke(stypy.reporting.localization.Localization(__file__, 176, 13), exit___354516, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'cheb_odd' (line 179)
        cheb_odd_354519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'cheb_odd', False)
        # Getting the type of 'cheb_odd_true' (line 179)
        cheb_odd_true_354520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'cheb_odd_true', False)
        # Processing the call keyword arguments (line 179)
        int_354521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 67), 'int')
        keyword_354522 = int_354521
        kwargs_354523 = {'decimal': keyword_354522}
        # Getting the type of 'assert_array_almost_equal' (line 179)
        assert_array_almost_equal_354518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 179)
        assert_array_almost_equal_call_result_354524 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), assert_array_almost_equal_354518, *[cheb_odd_354519, cheb_odd_true_354520], **kwargs_354523)
        
        
        # ################# End of 'test_cheb_odd_high_attenuation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_odd_high_attenuation' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_354525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354525)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_odd_high_attenuation'
        return stypy_return_type_354525


    @norecursion
    def test_cheb_even_high_attenuation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_even_high_attenuation'
        module_type_store = module_type_store.open_function_context('test_cheb_even_high_attenuation', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_localization', localization)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_function_name', 'TestChebWin.test_cheb_even_high_attenuation')
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_param_names_list', [])
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChebWin.test_cheb_even_high_attenuation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.test_cheb_even_high_attenuation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_even_high_attenuation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_even_high_attenuation(...)' code ##################

        
        # Call to suppress_warnings(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_354527 = {}
        # Getting the type of 'suppress_warnings' (line 182)
        suppress_warnings_354526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 182)
        suppress_warnings_call_result_354528 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), suppress_warnings_354526, *[], **kwargs_354527)
        
        with_354529 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 182, 13), suppress_warnings_call_result_354528, 'with parameter', '__enter__', '__exit__')

        if with_354529:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 182)
            enter___354530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), suppress_warnings_call_result_354528, '__enter__')
            with_enter_354531 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), enter___354530)
            # Assigning a type to the variable 'sup' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'sup', with_enter_354531)
            
            # Call to filter(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 'UserWarning' (line 183)
            UserWarning_354534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'UserWarning', False)
            str_354535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 183)
            kwargs_354536 = {}
            # Getting the type of 'sup' (line 183)
            sup_354532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 183)
            filter_354533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), sup_354532, 'filter')
            # Calling filter(args, kwargs) (line 183)
            filter_call_result_354537 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), filter_354533, *[UserWarning_354534, str_354535], **kwargs_354536)
            
            
            # Assigning a Call to a Name (line 184):
            
            # Call to chebwin(...): (line 184)
            # Processing the call arguments (line 184)
            int_354540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'int')
            # Processing the call keyword arguments (line 184)
            int_354541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 46), 'int')
            keyword_354542 = int_354541
            kwargs_354543 = {'at': keyword_354542}
            # Getting the type of 'signal' (line 184)
            signal_354538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 184)
            chebwin_354539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 24), signal_354538, 'chebwin')
            # Calling chebwin(args, kwargs) (line 184)
            chebwin_call_result_354544 = invoke(stypy.reporting.localization.Localization(__file__, 184, 24), chebwin_354539, *[int_354540], **kwargs_354543)
            
            # Assigning a type to the variable 'cheb_even' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'cheb_even', chebwin_call_result_354544)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 182)
            exit___354545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), suppress_warnings_call_result_354528, '__exit__')
            with_exit_354546 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), exit___354545, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'cheb_even' (line 185)
        cheb_even_354548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 'cheb_even', False)
        # Getting the type of 'cheb_even_true' (line 185)
        cheb_even_true_354549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 45), 'cheb_even_true', False)
        # Processing the call keyword arguments (line 185)
        int_354550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 69), 'int')
        keyword_354551 = int_354550
        kwargs_354552 = {'decimal': keyword_354551}
        # Getting the type of 'assert_array_almost_equal' (line 185)
        assert_array_almost_equal_354547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 185)
        assert_array_almost_equal_call_result_354553 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_array_almost_equal_354547, *[cheb_even_354548, cheb_even_true_354549], **kwargs_354552)
        
        
        # ################# End of 'test_cheb_even_high_attenuation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_even_high_attenuation' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_354554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354554)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_even_high_attenuation'
        return stypy_return_type_354554


    @norecursion
    def test_cheb_odd_low_attenuation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_odd_low_attenuation'
        module_type_store = module_type_store.open_function_context('test_cheb_odd_low_attenuation', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_localization', localization)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_function_name', 'TestChebWin.test_cheb_odd_low_attenuation')
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_param_names_list', [])
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChebWin.test_cheb_odd_low_attenuation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.test_cheb_odd_low_attenuation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_odd_low_attenuation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_odd_low_attenuation(...)' code ##################

        
        # Assigning a Call to a Name (line 188):
        
        # Call to array(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Obtaining an instance of the builtin type 'list' (line 188)
        list_354556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 188)
        # Adding element type (line 188)
        float_354557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354557)
        # Adding element type (line 188)
        float_354558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354558)
        # Adding element type (line 188)
        float_354559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354559)
        # Adding element type (line 188)
        float_354560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354560)
        # Adding element type (line 188)
        float_354561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354561)
        # Adding element type (line 188)
        float_354562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354562)
        # Adding element type (line 188)
        float_354563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 37), list_354556, float_354563)
        
        # Processing the call keyword arguments (line 188)
        kwargs_354564 = {}
        # Getting the type of 'array' (line 188)
        array_354555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'array', False)
        # Calling array(args, kwargs) (line 188)
        array_call_result_354565 = invoke(stypy.reporting.localization.Localization(__file__, 188, 31), array_354555, *[list_354556], **kwargs_354564)
        
        # Assigning a type to the variable 'cheb_odd_low_at_true' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'cheb_odd_low_at_true', array_call_result_354565)
        
        # Call to suppress_warnings(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_354567 = {}
        # Getting the type of 'suppress_warnings' (line 191)
        suppress_warnings_354566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 191)
        suppress_warnings_call_result_354568 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), suppress_warnings_354566, *[], **kwargs_354567)
        
        with_354569 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 191, 13), suppress_warnings_call_result_354568, 'with parameter', '__enter__', '__exit__')

        if with_354569:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 191)
            enter___354570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 13), suppress_warnings_call_result_354568, '__enter__')
            with_enter_354571 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), enter___354570)
            # Assigning a type to the variable 'sup' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'sup', with_enter_354571)
            
            # Call to filter(...): (line 192)
            # Processing the call arguments (line 192)
            # Getting the type of 'UserWarning' (line 192)
            UserWarning_354574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'UserWarning', False)
            str_354575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 192)
            kwargs_354576 = {}
            # Getting the type of 'sup' (line 192)
            sup_354572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 192)
            filter_354573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), sup_354572, 'filter')
            # Calling filter(args, kwargs) (line 192)
            filter_call_result_354577 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), filter_354573, *[UserWarning_354574, str_354575], **kwargs_354576)
            
            
            # Assigning a Call to a Name (line 193):
            
            # Call to chebwin(...): (line 193)
            # Processing the call arguments (line 193)
            int_354580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'int')
            # Processing the call keyword arguments (line 193)
            int_354581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'int')
            keyword_354582 = int_354581
            kwargs_354583 = {'at': keyword_354582}
            # Getting the type of 'signal' (line 193)
            signal_354578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 193)
            chebwin_354579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), signal_354578, 'chebwin')
            # Calling chebwin(args, kwargs) (line 193)
            chebwin_call_result_354584 = invoke(stypy.reporting.localization.Localization(__file__, 193, 23), chebwin_354579, *[int_354580], **kwargs_354583)
            
            # Assigning a type to the variable 'cheb_odd' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'cheb_odd', chebwin_call_result_354584)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 191)
            exit___354585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 13), suppress_warnings_call_result_354568, '__exit__')
            with_exit_354586 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), exit___354585, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'cheb_odd' (line 194)
        cheb_odd_354588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'cheb_odd', False)
        # Getting the type of 'cheb_odd_low_at_true' (line 194)
        cheb_odd_low_at_true_354589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 44), 'cheb_odd_low_at_true', False)
        # Processing the call keyword arguments (line 194)
        int_354590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 74), 'int')
        keyword_354591 = int_354590
        kwargs_354592 = {'decimal': keyword_354591}
        # Getting the type of 'assert_array_almost_equal' (line 194)
        assert_array_almost_equal_354587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 194)
        assert_array_almost_equal_call_result_354593 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assert_array_almost_equal_354587, *[cheb_odd_354588, cheb_odd_low_at_true_354589], **kwargs_354592)
        
        
        # ################# End of 'test_cheb_odd_low_attenuation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_odd_low_attenuation' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_354594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354594)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_odd_low_attenuation'
        return stypy_return_type_354594


    @norecursion
    def test_cheb_even_low_attenuation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_even_low_attenuation'
        module_type_store = module_type_store.open_function_context('test_cheb_even_low_attenuation', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_localization', localization)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_function_name', 'TestChebWin.test_cheb_even_low_attenuation')
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_param_names_list', [])
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChebWin.test_cheb_even_low_attenuation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.test_cheb_even_low_attenuation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_even_low_attenuation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_even_low_attenuation(...)' code ##################

        
        # Assigning a Call to a Name (line 197):
        
        # Call to array(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_354596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_354597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354597)
        # Adding element type (line 197)
        float_354598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354598)
        # Adding element type (line 197)
        float_354599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354599)
        # Adding element type (line 197)
        float_354600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354600)
        # Adding element type (line 197)
        float_354601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354601)
        # Adding element type (line 197)
        float_354602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354602)
        # Adding element type (line 197)
        float_354603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354603)
        # Adding element type (line 197)
        float_354604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 38), list_354596, float_354604)
        
        # Processing the call keyword arguments (line 197)
        kwargs_354605 = {}
        # Getting the type of 'array' (line 197)
        array_354595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 32), 'array', False)
        # Calling array(args, kwargs) (line 197)
        array_call_result_354606 = invoke(stypy.reporting.localization.Localization(__file__, 197, 32), array_354595, *[list_354596], **kwargs_354605)
        
        # Assigning a type to the variable 'cheb_even_low_at_true' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'cheb_even_low_at_true', array_call_result_354606)
        
        # Call to suppress_warnings(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_354608 = {}
        # Getting the type of 'suppress_warnings' (line 200)
        suppress_warnings_354607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 200)
        suppress_warnings_call_result_354609 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), suppress_warnings_354607, *[], **kwargs_354608)
        
        with_354610 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 200, 13), suppress_warnings_call_result_354609, 'with parameter', '__enter__', '__exit__')

        if with_354610:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 200)
            enter___354611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 13), suppress_warnings_call_result_354609, '__enter__')
            with_enter_354612 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), enter___354611)
            # Assigning a type to the variable 'sup' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'sup', with_enter_354612)
            
            # Call to filter(...): (line 201)
            # Processing the call arguments (line 201)
            # Getting the type of 'UserWarning' (line 201)
            UserWarning_354615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'UserWarning', False)
            str_354616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 201)
            kwargs_354617 = {}
            # Getting the type of 'sup' (line 201)
            sup_354613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 201)
            filter_354614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), sup_354613, 'filter')
            # Calling filter(args, kwargs) (line 201)
            filter_call_result_354618 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), filter_354614, *[UserWarning_354615, str_354616], **kwargs_354617)
            
            
            # Assigning a Call to a Name (line 202):
            
            # Call to chebwin(...): (line 202)
            # Processing the call arguments (line 202)
            int_354621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'int')
            # Processing the call keyword arguments (line 202)
            int_354622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'int')
            keyword_354623 = int_354622
            kwargs_354624 = {'at': keyword_354623}
            # Getting the type of 'signal' (line 202)
            signal_354619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'signal', False)
            # Obtaining the member 'chebwin' of a type (line 202)
            chebwin_354620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 24), signal_354619, 'chebwin')
            # Calling chebwin(args, kwargs) (line 202)
            chebwin_call_result_354625 = invoke(stypy.reporting.localization.Localization(__file__, 202, 24), chebwin_354620, *[int_354621], **kwargs_354624)
            
            # Assigning a type to the variable 'cheb_even' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'cheb_even', chebwin_call_result_354625)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 200)
            exit___354626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 13), suppress_warnings_call_result_354609, '__exit__')
            with_exit_354627 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), exit___354626, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'cheb_even' (line 203)
        cheb_even_354629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), 'cheb_even', False)
        # Getting the type of 'cheb_even_low_at_true' (line 203)
        cheb_even_low_at_true_354630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 45), 'cheb_even_low_at_true', False)
        # Processing the call keyword arguments (line 203)
        int_354631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 76), 'int')
        keyword_354632 = int_354631
        kwargs_354633 = {'decimal': keyword_354632}
        # Getting the type of 'assert_array_almost_equal' (line 203)
        assert_array_almost_equal_354628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 203)
        assert_array_almost_equal_call_result_354634 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), assert_array_almost_equal_354628, *[cheb_even_354629, cheb_even_low_at_true_354630], **kwargs_354633)
        
        
        # ################# End of 'test_cheb_even_low_attenuation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_even_low_attenuation' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_354635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_even_low_attenuation'
        return stypy_return_type_354635


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 151, 0, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChebWin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestChebWin' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'TestChebWin', TestChebWin)

# Assigning a Dict to a Name (line 206):

# Obtaining an instance of the builtin type 'dict' (line 206)
dict_354636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 206)
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 207)
tuple_354637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 207)
# Adding element type (line 207)
int_354638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_354637, int_354638)
# Adding element type (line 207)
# Getting the type of 'None' (line 207)
None_354639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_354637, None_354639)
# Adding element type (line 207)
float_354640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_354637, float_354640)
# Adding element type (line 207)
# Getting the type of 'False' (line 207)
False_354641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 5), tuple_354637, False_354641)


# Call to array(...): (line 208)
# Processing the call arguments (line 208)

# Obtaining an instance of the builtin type 'list' (line 208)
list_354643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 208)
# Adding element type (line 208)
float_354644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 14), list_354643, float_354644)
# Adding element type (line 208)
float_354645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 14), list_354643, float_354645)
# Adding element type (line 208)
float_354646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 14), list_354643, float_354646)
# Adding element type (line 208)
float_354647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 14), list_354643, float_354647)

# Processing the call keyword arguments (line 208)
kwargs_354648 = {}
# Getting the type of 'array' (line 208)
array_354642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'array', False)
# Calling array(args, kwargs) (line 208)
array_call_result_354649 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), array_354642, *[list_354643], **kwargs_354648)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354637, array_call_result_354649))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 211)
tuple_354650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 211)
# Adding element type (line 211)
int_354651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_354650, int_354651)
# Adding element type (line 211)
# Getting the type of 'None' (line 211)
None_354652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_354650, None_354652)
# Adding element type (line 211)
float_354653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_354650, float_354653)
# Adding element type (line 211)
# Getting the type of 'True' (line 211)
True_354654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 5), tuple_354650, True_354654)


# Call to array(...): (line 211)
# Processing the call arguments (line 211)

# Obtaining an instance of the builtin type 'list' (line 211)
list_354656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 211)
# Adding element type (line 211)
float_354657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_354656, float_354657)
# Adding element type (line 211)
float_354658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_354656, float_354658)
# Adding element type (line 211)
float_354659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_354656, float_354659)
# Adding element type (line 211)
float_354660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_354656, float_354660)

# Processing the call keyword arguments (line 211)
kwargs_354661 = {}
# Getting the type of 'array' (line 211)
array_354655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'array', False)
# Calling array(args, kwargs) (line 211)
array_call_result_354662 = invoke(stypy.reporting.localization.Localization(__file__, 211, 26), array_354655, *[list_354656], **kwargs_354661)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354650, array_call_result_354662))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 213)
tuple_354663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 213)
# Adding element type (line 213)
int_354664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 5), tuple_354663, int_354664)
# Adding element type (line 213)
# Getting the type of 'None' (line 213)
None_354665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 5), tuple_354663, None_354665)
# Adding element type (line 213)
float_354666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 5), tuple_354663, float_354666)
# Adding element type (line 213)
# Getting the type of 'False' (line 213)
False_354667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 5), tuple_354663, False_354667)


# Call to array(...): (line 213)
# Processing the call arguments (line 213)

# Obtaining an instance of the builtin type 'list' (line 213)
list_354669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 213)
# Adding element type (line 213)
float_354670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 33), list_354669, float_354670)
# Adding element type (line 213)
float_354671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 33), list_354669, float_354671)
# Adding element type (line 213)
float_354672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 75), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 33), list_354669, float_354672)
# Adding element type (line 213)
float_354673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 33), list_354669, float_354673)

# Processing the call keyword arguments (line 213)
kwargs_354674 = {}
# Getting the type of 'array' (line 213)
array_354668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'array', False)
# Calling array(args, kwargs) (line 213)
array_call_result_354675 = invoke(stypy.reporting.localization.Localization(__file__, 213, 27), array_354668, *[list_354669], **kwargs_354674)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354663, array_call_result_354675))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 215)
tuple_354676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 215)
# Adding element type (line 215)
int_354677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 5), tuple_354676, int_354677)
# Adding element type (line 215)
# Getting the type of 'None' (line 215)
None_354678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 5), tuple_354676, None_354678)
# Adding element type (line 215)
float_354679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 5), tuple_354676, float_354679)
# Adding element type (line 215)
# Getting the type of 'True' (line 215)
True_354680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 5), tuple_354676, True_354680)


# Call to array(...): (line 215)
# Processing the call arguments (line 215)

# Obtaining an instance of the builtin type 'list' (line 215)
list_354682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 215)
# Adding element type (line 215)
float_354683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 32), list_354682, float_354683)
# Adding element type (line 215)
float_354684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 32), list_354682, float_354684)
# Adding element type (line 215)
float_354685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 32), list_354682, float_354685)
# Adding element type (line 215)
float_354686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 32), list_354682, float_354686)

# Processing the call keyword arguments (line 215)
kwargs_354687 = {}
# Getting the type of 'array' (line 215)
array_354681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 26), 'array', False)
# Calling array(args, kwargs) (line 215)
array_call_result_354688 = invoke(stypy.reporting.localization.Localization(__file__, 215, 26), array_354681, *[list_354682], **kwargs_354687)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354676, array_call_result_354688))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 217)
tuple_354689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 217)
# Adding element type (line 217)
int_354690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 5), tuple_354689, int_354690)
# Adding element type (line 217)
int_354691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 5), tuple_354689, int_354691)
# Adding element type (line 217)
float_354692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 5), tuple_354689, float_354692)
# Adding element type (line 217)
# Getting the type of 'False' (line 217)
False_354693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 5), tuple_354689, False_354693)


# Call to array(...): (line 218)
# Processing the call arguments (line 218)

# Obtaining an instance of the builtin type 'list' (line 218)
list_354695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 218)
# Adding element type (line 218)
float_354696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 14), list_354695, float_354696)
# Adding element type (line 218)
float_354697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 14), list_354695, float_354697)
# Adding element type (line 218)
float_354698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 14), list_354695, float_354698)
# Adding element type (line 218)
float_354699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 14), list_354695, float_354699)

# Processing the call keyword arguments (line 218)
kwargs_354700 = {}
# Getting the type of 'array' (line 218)
array_354694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'array', False)
# Calling array(args, kwargs) (line 218)
array_call_result_354701 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), array_354694, *[list_354695], **kwargs_354700)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354689, array_call_result_354701))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 220)
tuple_354702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 220)
# Adding element type (line 220)
int_354703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 5), tuple_354702, int_354703)
# Adding element type (line 220)
int_354704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 5), tuple_354702, int_354704)
# Adding element type (line 220)
float_354705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 5), tuple_354702, float_354705)
# Adding element type (line 220)
# Getting the type of 'True' (line 220)
True_354706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 220, 5), tuple_354702, True_354706)

# Getting the type of 'None' (line 220)
None_354707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354702, None_354707))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 221)
tuple_354708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 221)
# Adding element type (line 221)
int_354709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 5), tuple_354708, int_354709)
# Adding element type (line 221)
int_354710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 5), tuple_354708, int_354710)
# Adding element type (line 221)
float_354711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 5), tuple_354708, float_354711)
# Adding element type (line 221)
# Getting the type of 'False' (line 221)
False_354712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 5), tuple_354708, False_354712)


# Call to array(...): (line 221)
# Processing the call arguments (line 221)

# Obtaining an instance of the builtin type 'list' (line 221)
list_354714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 221)
# Adding element type (line 221)
float_354715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 30), list_354714, float_354715)
# Adding element type (line 221)
float_354716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 51), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 30), list_354714, float_354716)
# Adding element type (line 221)
float_354717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 72), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 30), list_354714, float_354717)
# Adding element type (line 221)
float_354718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 30), list_354714, float_354718)

# Processing the call keyword arguments (line 221)
kwargs_354719 = {}
# Getting the type of 'array' (line 221)
array_354713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'array', False)
# Calling array(args, kwargs) (line 221)
array_call_result_354720 = invoke(stypy.reporting.localization.Localization(__file__, 221, 24), array_354713, *[list_354714], **kwargs_354719)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354708, array_call_result_354720))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 223)
tuple_354721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 223)
# Adding element type (line 223)
int_354722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 5), tuple_354721, int_354722)
# Adding element type (line 223)
int_354723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 5), tuple_354721, int_354723)
# Adding element type (line 223)
float_354724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 5), tuple_354721, float_354724)
# Adding element type (line 223)
# Getting the type of 'True' (line 223)
True_354725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 5), tuple_354721, True_354725)

# Getting the type of 'None' (line 223)
None_354726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354721, None_354726))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 224)
tuple_354727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 224)
# Adding element type (line 224)
int_354728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 5), tuple_354727, int_354728)
# Adding element type (line 224)
# Getting the type of 'None' (line 224)
None_354729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 5), tuple_354727, None_354729)
# Adding element type (line 224)
float_354730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 5), tuple_354727, float_354730)
# Adding element type (line 224)
# Getting the type of 'True' (line 224)
True_354731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 5), tuple_354727, True_354731)


# Call to array(...): (line 225)
# Processing the call arguments (line 225)

# Obtaining an instance of the builtin type 'list' (line 225)
list_354733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 225)
# Adding element type (line 225)
float_354734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 14), list_354733, float_354734)
# Adding element type (line 225)
float_354735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 14), list_354733, float_354735)
# Adding element type (line 225)
float_354736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 14), list_354733, float_354736)
# Adding element type (line 225)
float_354737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 14), list_354733, float_354737)
# Adding element type (line 225)
float_354738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 40), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 14), list_354733, float_354738)

# Processing the call keyword arguments (line 225)
kwargs_354739 = {}
# Getting the type of 'array' (line 225)
array_354732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'array', False)
# Calling array(args, kwargs) (line 225)
array_call_result_354740 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), array_354732, *[list_354733], **kwargs_354739)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354727, array_call_result_354740))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 228)
tuple_354741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 228)
# Adding element type (line 228)
int_354742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 5), tuple_354741, int_354742)
# Adding element type (line 228)
# Getting the type of 'None' (line 228)
None_354743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 5), tuple_354741, None_354743)
# Adding element type (line 228)
float_354744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 5), tuple_354741, float_354744)
# Adding element type (line 228)
# Getting the type of 'True' (line 228)
True_354745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 5), tuple_354741, True_354745)


# Call to array(...): (line 228)
# Processing the call arguments (line 228)

# Obtaining an instance of the builtin type 'list' (line 228)
list_354747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 228)
# Adding element type (line 228)
float_354748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), list_354747, float_354748)
# Adding element type (line 228)
float_354749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), list_354747, float_354749)
# Adding element type (line 228)
float_354750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 74), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), list_354747, float_354750)
# Adding element type (line 228)
float_354751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), list_354747, float_354751)
# Adding element type (line 228)
float_354752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), list_354747, float_354752)

# Processing the call keyword arguments (line 228)
kwargs_354753 = {}
# Getting the type of 'array' (line 228)
array_354746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'array', False)
# Calling array(args, kwargs) (line 228)
array_call_result_354754 = invoke(stypy.reporting.localization.Localization(__file__, 228, 26), array_354746, *[list_354747], **kwargs_354753)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354741, array_call_result_354754))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 230)
tuple_354755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 230)
# Adding element type (line 230)
int_354756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 5), tuple_354755, int_354756)
# Adding element type (line 230)
int_354757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 5), tuple_354755, int_354757)
# Adding element type (line 230)
float_354758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 5), tuple_354755, float_354758)
# Adding element type (line 230)
# Getting the type of 'True' (line 230)
True_354759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 5), tuple_354755, True_354759)

# Getting the type of 'None' (line 230)
None_354760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354755, None_354760))
# Adding element type (key, value) (line 206)

# Obtaining an instance of the builtin type 'tuple' (line 231)
tuple_354761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 231)
# Adding element type (line 231)
int_354762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 5), tuple_354761, int_354762)
# Adding element type (line 231)
int_354763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 5), tuple_354761, int_354763)
# Adding element type (line 231)
float_354764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 5), tuple_354761, float_354764)
# Adding element type (line 231)
# Getting the type of 'True' (line 231)
True_354765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 5), tuple_354761, True_354765)

# Getting the type of 'None' (line 231)
None_354766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 19), dict_354636, (tuple_354761, None_354766))

# Assigning a type to the variable 'exponential_data' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'exponential_data', dict_354636)

@norecursion
def test_exponential(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_exponential'
    module_type_store = module_type_store.open_function_context('test_exponential', 235, 0, False)
    
    # Passed parameters checking function
    test_exponential.stypy_localization = localization
    test_exponential.stypy_type_of_self = None
    test_exponential.stypy_type_store = module_type_store
    test_exponential.stypy_function_name = 'test_exponential'
    test_exponential.stypy_param_names_list = []
    test_exponential.stypy_varargs_param_name = None
    test_exponential.stypy_kwargs_param_name = None
    test_exponential.stypy_call_defaults = defaults
    test_exponential.stypy_call_varargs = varargs
    test_exponential.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_exponential', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_exponential', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_exponential(...)' code ##################

    
    
    # Call to items(...): (line 236)
    # Processing the call keyword arguments (line 236)
    kwargs_354769 = {}
    # Getting the type of 'exponential_data' (line 236)
    exponential_data_354767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'exponential_data', False)
    # Obtaining the member 'items' of a type (line 236)
    items_354768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), exponential_data_354767, 'items')
    # Calling items(args, kwargs) (line 236)
    items_call_result_354770 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), items_354768, *[], **kwargs_354769)
    
    # Testing the type of a for loop iterable (line 236)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 236, 4), items_call_result_354770)
    # Getting the type of the for loop variable (line 236)
    for_loop_var_354771 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 236, 4), items_call_result_354770)
    # Assigning a type to the variable 'k' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 4), for_loop_var_354771))
    # Assigning a type to the variable 'v' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 4), for_loop_var_354771))
    # SSA begins for a for statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 237)
    # Getting the type of 'v' (line 237)
    v_354772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'v')
    # Getting the type of 'None' (line 237)
    None_354773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'None')
    
    (may_be_354774, more_types_in_union_354775) = may_be_none(v_354772, None_354773)

    if may_be_354774:

        if more_types_in_union_354775:
            # Runtime conditional SSA (line 237)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to assert_raises(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'ValueError' (line 238)
        ValueError_354777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'ValueError', False)
        # Getting the type of 'signal' (line 238)
        signal_354778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 38), 'signal', False)
        # Obtaining the member 'exponential' of a type (line 238)
        exponential_354779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 38), signal_354778, 'exponential')
        # Getting the type of 'k' (line 238)
        k_354780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 59), 'k', False)
        # Processing the call keyword arguments (line 238)
        kwargs_354781 = {}
        # Getting the type of 'assert_raises' (line 238)
        assert_raises_354776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 238)
        assert_raises_call_result_354782 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), assert_raises_354776, *[ValueError_354777, exponential_354779, k_354780], **kwargs_354781)
        

        if more_types_in_union_354775:
            # Runtime conditional SSA for else branch (line 237)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_354774) or more_types_in_union_354775):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to exponential(...): (line 240)
        # Getting the type of 'k' (line 240)
        k_354785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 38), 'k', False)
        # Processing the call keyword arguments (line 240)
        kwargs_354786 = {}
        # Getting the type of 'signal' (line 240)
        signal_354783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'signal', False)
        # Obtaining the member 'exponential' of a type (line 240)
        exponential_354784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 18), signal_354783, 'exponential')
        # Calling exponential(args, kwargs) (line 240)
        exponential_call_result_354787 = invoke(stypy.reporting.localization.Localization(__file__, 240, 18), exponential_354784, *[k_354785], **kwargs_354786)
        
        # Assigning a type to the variable 'win' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'win', exponential_call_result_354787)
        
        # Call to assert_allclose(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'win' (line 241)
        win_354789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 28), 'win', False)
        # Getting the type of 'v' (line 241)
        v_354790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'v', False)
        # Processing the call keyword arguments (line 241)
        float_354791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 41), 'float')
        keyword_354792 = float_354791
        kwargs_354793 = {'rtol': keyword_354792}
        # Getting the type of 'assert_allclose' (line 241)
        assert_allclose_354788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 241)
        assert_allclose_call_result_354794 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), assert_allclose_354788, *[win_354789, v_354790], **kwargs_354793)
        

        if (may_be_354774 and more_types_in_union_354775):
            # SSA join for if statement (line 237)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_exponential(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_exponential' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_354795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_354795)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_exponential'
    return stypy_return_type_354795

# Assigning a type to the variable 'test_exponential' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'test_exponential', test_exponential)
# Declaration of the 'TestFlatTop' class

class TestFlatTop(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_function_name', 'TestFlatTop.test_basic')
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFlatTop.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFlatTop.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to flattop(...): (line 247)
        # Processing the call arguments (line 247)
        int_354799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 39), 'int')
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'False' (line 247)
        False_354800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'False', False)
        keyword_354801 = False_354800
        kwargs_354802 = {'sym': keyword_354801}
        # Getting the type of 'signal' (line 247)
        signal_354797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'signal', False)
        # Obtaining the member 'flattop' of a type (line 247)
        flattop_354798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 24), signal_354797, 'flattop')
        # Calling flattop(args, kwargs) (line 247)
        flattop_call_result_354803 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), flattop_354798, *[int_354799], **kwargs_354802)
        
        
        # Obtaining an instance of the builtin type 'list' (line 248)
        list_354804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 248)
        # Adding element type (line 248)
        float_354805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354805)
        # Adding element type (line 248)
        float_354806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354806)
        # Adding element type (line 248)
        float_354807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354807)
        # Adding element type (line 248)
        float_354808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354808)
        # Adding element type (line 248)
        float_354809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354809)
        # Adding element type (line 248)
        float_354810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 24), list_354804, float_354810)
        
        # Processing the call keyword arguments (line 247)
        kwargs_354811 = {}
        # Getting the type of 'assert_allclose' (line 247)
        assert_allclose_354796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 247)
        assert_allclose_call_result_354812 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_allclose_354796, *[flattop_call_result_354803, list_354804], **kwargs_354811)
        
        
        # Call to assert_allclose(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to flattop(...): (line 250)
        # Processing the call arguments (line 250)
        int_354816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 39), 'int')
        # Processing the call keyword arguments (line 250)
        # Getting the type of 'False' (line 250)
        False_354817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'False', False)
        keyword_354818 = False_354817
        kwargs_354819 = {'sym': keyword_354818}
        # Getting the type of 'signal' (line 250)
        signal_354814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'signal', False)
        # Obtaining the member 'flattop' of a type (line 250)
        flattop_354815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), signal_354814, 'flattop')
        # Calling flattop(args, kwargs) (line 250)
        flattop_call_result_354820 = invoke(stypy.reporting.localization.Localization(__file__, 250, 24), flattop_354815, *[int_354816], **kwargs_354819)
        
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_354821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        # Adding element type (line 251)
        float_354822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354822)
        # Adding element type (line 251)
        float_354823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354823)
        # Adding element type (line 251)
        float_354824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354824)
        # Adding element type (line 251)
        float_354825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354825)
        # Adding element type (line 251)
        float_354826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354826)
        # Adding element type (line 251)
        float_354827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354827)
        # Adding element type (line 251)
        float_354828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 24), list_354821, float_354828)
        
        # Processing the call keyword arguments (line 250)
        kwargs_354829 = {}
        # Getting the type of 'assert_allclose' (line 250)
        assert_allclose_354813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 250)
        assert_allclose_call_result_354830 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), assert_allclose_354813, *[flattop_call_result_354820, list_354821], **kwargs_354829)
        
        
        # Call to assert_allclose(...): (line 255)
        # Processing the call arguments (line 255)
        
        # Call to flattop(...): (line 255)
        # Processing the call arguments (line 255)
        int_354834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 39), 'int')
        # Processing the call keyword arguments (line 255)
        kwargs_354835 = {}
        # Getting the type of 'signal' (line 255)
        signal_354832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'signal', False)
        # Obtaining the member 'flattop' of a type (line 255)
        flattop_354833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), signal_354832, 'flattop')
        # Calling flattop(args, kwargs) (line 255)
        flattop_call_result_354836 = invoke(stypy.reporting.localization.Localization(__file__, 255, 24), flattop_354833, *[int_354834], **kwargs_354835)
        
        
        # Obtaining an instance of the builtin type 'list' (line 256)
        list_354837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 256)
        # Adding element type (line 256)
        float_354838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354838)
        # Adding element type (line 256)
        float_354839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354839)
        # Adding element type (line 256)
        float_354840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354840)
        # Adding element type (line 256)
        float_354841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354841)
        # Adding element type (line 256)
        float_354842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354842)
        # Adding element type (line 256)
        float_354843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 24), list_354837, float_354843)
        
        # Processing the call keyword arguments (line 255)
        kwargs_354844 = {}
        # Getting the type of 'assert_allclose' (line 255)
        assert_allclose_354831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 255)
        assert_allclose_call_result_354845 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assert_allclose_354831, *[flattop_call_result_354836, list_354837], **kwargs_354844)
        
        
        # Call to assert_allclose(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Call to flattop(...): (line 259)
        # Processing the call arguments (line 259)
        int_354849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 39), 'int')
        # Getting the type of 'True' (line 259)
        True_354850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 42), 'True', False)
        # Processing the call keyword arguments (line 259)
        kwargs_354851 = {}
        # Getting the type of 'signal' (line 259)
        signal_354847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'signal', False)
        # Obtaining the member 'flattop' of a type (line 259)
        flattop_354848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), signal_354847, 'flattop')
        # Calling flattop(args, kwargs) (line 259)
        flattop_call_result_354852 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), flattop_354848, *[int_354849, True_354850], **kwargs_354851)
        
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_354853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        float_354854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354854)
        # Adding element type (line 260)
        float_354855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354855)
        # Adding element type (line 260)
        float_354856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354856)
        # Adding element type (line 260)
        float_354857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354857)
        # Adding element type (line 260)
        float_354858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354858)
        # Adding element type (line 260)
        float_354859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354859)
        # Adding element type (line 260)
        float_354860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 24), list_354853, float_354860)
        
        # Processing the call keyword arguments (line 259)
        kwargs_354861 = {}
        # Getting the type of 'assert_allclose' (line 259)
        assert_allclose_354846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 259)
        assert_allclose_call_result_354862 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), assert_allclose_354846, *[flattop_call_result_354852, list_354853], **kwargs_354861)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_354863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354863


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 244, 0, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFlatTop.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFlatTop' (line 244)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 0), 'TestFlatTop', TestFlatTop)
# Declaration of the 'TestGaussian' class

class TestGaussian(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGaussian.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_function_name', 'TestGaussian.test_basic')
        TestGaussian.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestGaussian.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGaussian.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussian.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Call to gaussian(...): (line 267)
        # Processing the call arguments (line 267)
        int_354867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 40), 'int')
        float_354868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 43), 'float')
        # Processing the call keyword arguments (line 267)
        kwargs_354869 = {}
        # Getting the type of 'signal' (line 267)
        signal_354865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'signal', False)
        # Obtaining the member 'gaussian' of a type (line 267)
        gaussian_354866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 24), signal_354865, 'gaussian')
        # Calling gaussian(args, kwargs) (line 267)
        gaussian_call_result_354870 = invoke(stypy.reporting.localization.Localization(__file__, 267, 24), gaussian_354866, *[int_354867, float_354868], **kwargs_354869)
        
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_354871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_354872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354872)
        # Adding element type (line 268)
        float_354873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354873)
        # Adding element type (line 268)
        float_354874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354874)
        # Adding element type (line 268)
        float_354875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354875)
        # Adding element type (line 268)
        float_354876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354876)
        # Adding element type (line 268)
        float_354877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 24), list_354871, float_354877)
        
        # Processing the call keyword arguments (line 267)
        kwargs_354878 = {}
        # Getting the type of 'assert_allclose' (line 267)
        assert_allclose_354864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 267)
        assert_allclose_call_result_354879 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assert_allclose_354864, *[gaussian_call_result_354870, list_354871], **kwargs_354878)
        
        
        # Call to assert_allclose(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Call to gaussian(...): (line 271)
        # Processing the call arguments (line 271)
        int_354883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 40), 'int')
        float_354884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 43), 'float')
        # Processing the call keyword arguments (line 271)
        kwargs_354885 = {}
        # Getting the type of 'signal' (line 271)
        signal_354881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'signal', False)
        # Obtaining the member 'gaussian' of a type (line 271)
        gaussian_354882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 24), signal_354881, 'gaussian')
        # Calling gaussian(args, kwargs) (line 271)
        gaussian_call_result_354886 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), gaussian_354882, *[int_354883, float_354884], **kwargs_354885)
        
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_354887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        float_354888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354888)
        # Adding element type (line 272)
        float_354889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354889)
        # Adding element type (line 272)
        float_354890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354890)
        # Adding element type (line 272)
        float_354891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354891)
        # Adding element type (line 272)
        float_354892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354892)
        # Adding element type (line 272)
        float_354893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354893)
        # Adding element type (line 272)
        float_354894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 24), list_354887, float_354894)
        
        # Processing the call keyword arguments (line 271)
        kwargs_354895 = {}
        # Getting the type of 'assert_allclose' (line 271)
        assert_allclose_354880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 271)
        assert_allclose_call_result_354896 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), assert_allclose_354880, *[gaussian_call_result_354886, list_354887], **kwargs_354895)
        
        
        # Call to assert_allclose(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to gaussian(...): (line 275)
        # Processing the call arguments (line 275)
        int_354900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 40), 'int')
        int_354901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 43), 'int')
        # Processing the call keyword arguments (line 275)
        kwargs_354902 = {}
        # Getting the type of 'signal' (line 275)
        signal_354898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'signal', False)
        # Obtaining the member 'gaussian' of a type (line 275)
        gaussian_354899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 24), signal_354898, 'gaussian')
        # Calling gaussian(args, kwargs) (line 275)
        gaussian_call_result_354903 = invoke(stypy.reporting.localization.Localization(__file__, 275, 24), gaussian_354899, *[int_354900, int_354901], **kwargs_354902)
        
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_354904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        float_354905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354905)
        # Adding element type (line 276)
        float_354906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354906)
        # Adding element type (line 276)
        float_354907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354907)
        # Adding element type (line 276)
        float_354908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354908)
        # Adding element type (line 276)
        float_354909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354909)
        # Adding element type (line 276)
        float_354910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354910)
        # Adding element type (line 276)
        float_354911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 24), list_354904, float_354911)
        
        # Processing the call keyword arguments (line 275)
        kwargs_354912 = {}
        # Getting the type of 'assert_allclose' (line 275)
        assert_allclose_354897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 275)
        assert_allclose_call_result_354913 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), assert_allclose_354897, *[gaussian_call_result_354903, list_354904], **kwargs_354912)
        
        
        # Call to assert_allclose(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Call to gaussian(...): (line 279)
        # Processing the call arguments (line 279)
        int_354917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 40), 'int')
        int_354918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 43), 'int')
        # Getting the type of 'False' (line 279)
        False_354919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 46), 'False', False)
        # Processing the call keyword arguments (line 279)
        kwargs_354920 = {}
        # Getting the type of 'signal' (line 279)
        signal_354915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'signal', False)
        # Obtaining the member 'gaussian' of a type (line 279)
        gaussian_354916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), signal_354915, 'gaussian')
        # Calling gaussian(args, kwargs) (line 279)
        gaussian_call_result_354921 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), gaussian_354916, *[int_354917, int_354918, False_354919], **kwargs_354920)
        
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_354922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        float_354923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354923)
        # Adding element type (line 280)
        float_354924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354924)
        # Adding element type (line 280)
        float_354925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354925)
        # Adding element type (line 280)
        float_354926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354926)
        # Adding element type (line 280)
        float_354927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354927)
        # Adding element type (line 280)
        float_354928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 24), list_354922, float_354928)
        
        # Processing the call keyword arguments (line 279)
        kwargs_354929 = {}
        # Getting the type of 'assert_allclose' (line 279)
        assert_allclose_354914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 279)
        assert_allclose_call_result_354930 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert_allclose_354914, *[gaussian_call_result_354921, list_354922], **kwargs_354929)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_354931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354931


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 264, 0, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussian.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGaussian' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'TestGaussian', TestGaussian)
# Declaration of the 'TestHamming' class

class TestHamming(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHamming.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestHamming.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHamming.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHamming.test_basic.__dict__.__setitem__('stypy_function_name', 'TestHamming.test_basic')
        TestHamming.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestHamming.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHamming.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHamming.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHamming.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHamming.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHamming.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHamming.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Call to hamming(...): (line 288)
        # Processing the call arguments (line 288)
        int_354935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 39), 'int')
        # Getting the type of 'False' (line 288)
        False_354936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 42), 'False', False)
        # Processing the call keyword arguments (line 288)
        kwargs_354937 = {}
        # Getting the type of 'signal' (line 288)
        signal_354933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'signal', False)
        # Obtaining the member 'hamming' of a type (line 288)
        hamming_354934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), signal_354933, 'hamming')
        # Calling hamming(args, kwargs) (line 288)
        hamming_call_result_354938 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), hamming_354934, *[int_354935, False_354936], **kwargs_354937)
        
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_354939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        float_354940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354940)
        # Adding element type (line 289)
        float_354941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354941)
        # Adding element type (line 289)
        float_354942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354942)
        # Adding element type (line 289)
        float_354943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354943)
        # Adding element type (line 289)
        float_354944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354944)
        # Adding element type (line 289)
        float_354945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 24), list_354939, float_354945)
        
        # Processing the call keyword arguments (line 288)
        kwargs_354946 = {}
        # Getting the type of 'assert_allclose' (line 288)
        assert_allclose_354932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 288)
        assert_allclose_call_result_354947 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), assert_allclose_354932, *[hamming_call_result_354938, list_354939], **kwargs_354946)
        
        
        # Call to assert_allclose(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Call to hamming(...): (line 290)
        # Processing the call arguments (line 290)
        int_354951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 39), 'int')
        # Processing the call keyword arguments (line 290)
        # Getting the type of 'False' (line 290)
        False_354952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 46), 'False', False)
        keyword_354953 = False_354952
        kwargs_354954 = {'sym': keyword_354953}
        # Getting the type of 'signal' (line 290)
        signal_354949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'signal', False)
        # Obtaining the member 'hamming' of a type (line 290)
        hamming_354950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), signal_354949, 'hamming')
        # Calling hamming(args, kwargs) (line 290)
        hamming_call_result_354955 = invoke(stypy.reporting.localization.Localization(__file__, 290, 24), hamming_354950, *[int_354951], **kwargs_354954)
        
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_354956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        # Adding element type (line 291)
        float_354957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354957)
        # Adding element type (line 291)
        float_354958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354958)
        # Adding element type (line 291)
        float_354959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354959)
        # Adding element type (line 291)
        float_354960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354960)
        # Adding element type (line 291)
        float_354961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354961)
        # Adding element type (line 291)
        float_354962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354962)
        # Adding element type (line 291)
        float_354963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 24), list_354956, float_354963)
        
        # Processing the call keyword arguments (line 290)
        kwargs_354964 = {}
        # Getting the type of 'assert_allclose' (line 290)
        assert_allclose_354948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 290)
        assert_allclose_call_result_354965 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), assert_allclose_354948, *[hamming_call_result_354955, list_354956], **kwargs_354964)
        
        
        # Call to assert_allclose(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to hamming(...): (line 294)
        # Processing the call arguments (line 294)
        int_354969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 39), 'int')
        # Processing the call keyword arguments (line 294)
        kwargs_354970 = {}
        # Getting the type of 'signal' (line 294)
        signal_354967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'signal', False)
        # Obtaining the member 'hamming' of a type (line 294)
        hamming_354968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), signal_354967, 'hamming')
        # Calling hamming(args, kwargs) (line 294)
        hamming_call_result_354971 = invoke(stypy.reporting.localization.Localization(__file__, 294, 24), hamming_354968, *[int_354969], **kwargs_354970)
        
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_354972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        float_354973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354973)
        # Adding element type (line 295)
        float_354974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354974)
        # Adding element type (line 295)
        float_354975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354975)
        # Adding element type (line 295)
        float_354976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354976)
        # Adding element type (line 295)
        float_354977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354977)
        # Adding element type (line 295)
        float_354978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 24), list_354972, float_354978)
        
        # Processing the call keyword arguments (line 294)
        kwargs_354979 = {}
        # Getting the type of 'assert_allclose' (line 294)
        assert_allclose_354966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 294)
        assert_allclose_call_result_354980 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), assert_allclose_354966, *[hamming_call_result_354971, list_354972], **kwargs_354979)
        
        
        # Call to assert_allclose(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Call to hamming(...): (line 297)
        # Processing the call arguments (line 297)
        int_354984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 39), 'int')
        # Processing the call keyword arguments (line 297)
        # Getting the type of 'True' (line 297)
        True_354985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'True', False)
        keyword_354986 = True_354985
        kwargs_354987 = {'sym': keyword_354986}
        # Getting the type of 'signal' (line 297)
        signal_354982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'signal', False)
        # Obtaining the member 'hamming' of a type (line 297)
        hamming_354983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 24), signal_354982, 'hamming')
        # Calling hamming(args, kwargs) (line 297)
        hamming_call_result_354988 = invoke(stypy.reporting.localization.Localization(__file__, 297, 24), hamming_354983, *[int_354984], **kwargs_354987)
        
        
        # Obtaining an instance of the builtin type 'list' (line 298)
        list_354989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 298)
        # Adding element type (line 298)
        float_354990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354990)
        # Adding element type (line 298)
        float_354991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354991)
        # Adding element type (line 298)
        float_354992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354992)
        # Adding element type (line 298)
        float_354993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354993)
        # Adding element type (line 298)
        float_354994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354994)
        # Adding element type (line 298)
        float_354995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354995)
        # Adding element type (line 298)
        float_354996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 24), list_354989, float_354996)
        
        # Processing the call keyword arguments (line 297)
        kwargs_354997 = {}
        # Getting the type of 'assert_allclose' (line 297)
        assert_allclose_354981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 297)
        assert_allclose_call_result_354998 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assert_allclose_354981, *[hamming_call_result_354988, list_354989], **kwargs_354997)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_354999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_354999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_354999


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 285, 0, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHamming.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHamming' (line 285)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'TestHamming', TestHamming)
# Declaration of the 'TestHann' class

class TestHann(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 303, 4, False)
        # Assigning a type to the variable 'self' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHann.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestHann.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHann.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHann.test_basic.__dict__.__setitem__('stypy_function_name', 'TestHann.test_basic')
        TestHann.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestHann.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHann.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHann.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHann.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHann.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHann.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHann.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 304)
        # Processing the call arguments (line 304)
        
        # Call to hann(...): (line 304)
        # Processing the call arguments (line 304)
        int_355003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 36), 'int')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'False' (line 304)
        False_355004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'False', False)
        keyword_355005 = False_355004
        kwargs_355006 = {'sym': keyword_355005}
        # Getting the type of 'signal' (line 304)
        signal_355001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'signal', False)
        # Obtaining the member 'hann' of a type (line 304)
        hann_355002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 24), signal_355001, 'hann')
        # Calling hann(args, kwargs) (line 304)
        hann_call_result_355007 = invoke(stypy.reporting.localization.Localization(__file__, 304, 24), hann_355002, *[int_355003], **kwargs_355006)
        
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_355008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        int_355009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, int_355009)
        # Adding element type (line 305)
        float_355010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, float_355010)
        # Adding element type (line 305)
        float_355011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, float_355011)
        # Adding element type (line 305)
        float_355012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, float_355012)
        # Adding element type (line 305)
        float_355013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, float_355013)
        # Adding element type (line 305)
        float_355014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 24), list_355008, float_355014)
        
        # Processing the call keyword arguments (line 304)
        kwargs_355015 = {}
        # Getting the type of 'assert_allclose' (line 304)
        assert_allclose_355000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 304)
        assert_allclose_call_result_355016 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assert_allclose_355000, *[hann_call_result_355007, list_355008], **kwargs_355015)
        
        
        # Call to assert_allclose(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Call to hann(...): (line 306)
        # Processing the call arguments (line 306)
        int_355020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 36), 'int')
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'False' (line 306)
        False_355021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 43), 'False', False)
        keyword_355022 = False_355021
        kwargs_355023 = {'sym': keyword_355022}
        # Getting the type of 'signal' (line 306)
        signal_355018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 24), 'signal', False)
        # Obtaining the member 'hann' of a type (line 306)
        hann_355019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 24), signal_355018, 'hann')
        # Calling hann(args, kwargs) (line 306)
        hann_call_result_355024 = invoke(stypy.reporting.localization.Localization(__file__, 306, 24), hann_355019, *[int_355020], **kwargs_355023)
        
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_355025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        int_355026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, int_355026)
        # Adding element type (line 307)
        float_355027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355027)
        # Adding element type (line 307)
        float_355028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355028)
        # Adding element type (line 307)
        float_355029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355029)
        # Adding element type (line 307)
        float_355030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355030)
        # Adding element type (line 307)
        float_355031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355031)
        # Adding element type (line 307)
        float_355032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 24), list_355025, float_355032)
        
        # Processing the call keyword arguments (line 306)
        kwargs_355033 = {}
        # Getting the type of 'assert_allclose' (line 306)
        assert_allclose_355017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 306)
        assert_allclose_call_result_355034 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), assert_allclose_355017, *[hann_call_result_355024, list_355025], **kwargs_355033)
        
        
        # Call to assert_allclose(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Call to hann(...): (line 310)
        # Processing the call arguments (line 310)
        int_355038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 36), 'int')
        # Getting the type of 'True' (line 310)
        True_355039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 39), 'True', False)
        # Processing the call keyword arguments (line 310)
        kwargs_355040 = {}
        # Getting the type of 'signal' (line 310)
        signal_355036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'signal', False)
        # Obtaining the member 'hann' of a type (line 310)
        hann_355037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 24), signal_355036, 'hann')
        # Calling hann(args, kwargs) (line 310)
        hann_call_result_355041 = invoke(stypy.reporting.localization.Localization(__file__, 310, 24), hann_355037, *[int_355038, True_355039], **kwargs_355040)
        
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_355042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        int_355043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, int_355043)
        # Adding element type (line 311)
        float_355044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, float_355044)
        # Adding element type (line 311)
        float_355045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, float_355045)
        # Adding element type (line 311)
        float_355046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, float_355046)
        # Adding element type (line 311)
        float_355047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, float_355047)
        # Adding element type (line 311)
        int_355048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 65), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 24), list_355042, int_355048)
        
        # Processing the call keyword arguments (line 310)
        kwargs_355049 = {}
        # Getting the type of 'assert_allclose' (line 310)
        assert_allclose_355035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 310)
        assert_allclose_call_result_355050 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), assert_allclose_355035, *[hann_call_result_355041, list_355042], **kwargs_355049)
        
        
        # Call to assert_allclose(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Call to hann(...): (line 313)
        # Processing the call arguments (line 313)
        int_355054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 36), 'int')
        # Processing the call keyword arguments (line 313)
        kwargs_355055 = {}
        # Getting the type of 'signal' (line 313)
        signal_355052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'signal', False)
        # Obtaining the member 'hann' of a type (line 313)
        hann_355053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 24), signal_355052, 'hann')
        # Calling hann(args, kwargs) (line 313)
        hann_call_result_355056 = invoke(stypy.reporting.localization.Localization(__file__, 313, 24), hann_355053, *[int_355054], **kwargs_355055)
        
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_355057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        int_355058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, int_355058)
        # Adding element type (line 314)
        float_355059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, float_355059)
        # Adding element type (line 314)
        float_355060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, float_355060)
        # Adding element type (line 314)
        float_355061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, float_355061)
        # Adding element type (line 314)
        float_355062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, float_355062)
        # Adding element type (line 314)
        float_355063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, float_355063)
        # Adding element type (line 314)
        int_355064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 24), list_355057, int_355064)
        
        # Processing the call keyword arguments (line 313)
        kwargs_355065 = {}
        # Getting the type of 'assert_allclose' (line 313)
        assert_allclose_355051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 313)
        assert_allclose_call_result_355066 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assert_allclose_355051, *[hann_call_result_355056, list_355057], **kwargs_355065)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 303)
        stypy_return_type_355067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355067


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 301, 0, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHann.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHann' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'TestHann', TestHann)
# Declaration of the 'TestKaiser' class

class TestKaiser(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKaiser.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_function_name', 'TestKaiser.test_basic')
        TestKaiser.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestKaiser.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKaiser.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKaiser.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Call to kaiser(...): (line 320)
        # Processing the call arguments (line 320)
        int_355071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 38), 'int')
        float_355072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 41), 'float')
        # Processing the call keyword arguments (line 320)
        kwargs_355073 = {}
        # Getting the type of 'signal' (line 320)
        signal_355069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 320)
        kaiser_355070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 24), signal_355069, 'kaiser')
        # Calling kaiser(args, kwargs) (line 320)
        kaiser_call_result_355074 = invoke(stypy.reporting.localization.Localization(__file__, 320, 24), kaiser_355070, *[int_355071, float_355072], **kwargs_355073)
        
        
        # Obtaining an instance of the builtin type 'list' (line 321)
        list_355075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 321)
        # Adding element type (line 321)
        float_355076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355076)
        # Adding element type (line 321)
        float_355077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355077)
        # Adding element type (line 321)
        float_355078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355078)
        # Adding element type (line 321)
        float_355079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355079)
        # Adding element type (line 321)
        float_355080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355080)
        # Adding element type (line 321)
        float_355081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 24), list_355075, float_355081)
        
        # Processing the call keyword arguments (line 320)
        kwargs_355082 = {}
        # Getting the type of 'assert_allclose' (line 320)
        assert_allclose_355068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 320)
        assert_allclose_call_result_355083 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), assert_allclose_355068, *[kaiser_call_result_355074, list_355075], **kwargs_355082)
        
        
        # Call to assert_allclose(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Call to kaiser(...): (line 324)
        # Processing the call arguments (line 324)
        int_355087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 38), 'int')
        float_355088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 41), 'float')
        # Processing the call keyword arguments (line 324)
        kwargs_355089 = {}
        # Getting the type of 'signal' (line 324)
        signal_355085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 324)
        kaiser_355086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 24), signal_355085, 'kaiser')
        # Calling kaiser(args, kwargs) (line 324)
        kaiser_call_result_355090 = invoke(stypy.reporting.localization.Localization(__file__, 324, 24), kaiser_355086, *[int_355087, float_355088], **kwargs_355089)
        
        
        # Obtaining an instance of the builtin type 'list' (line 325)
        list_355091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 325)
        # Adding element type (line 325)
        float_355092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355092)
        # Adding element type (line 325)
        float_355093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355093)
        # Adding element type (line 325)
        float_355094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355094)
        # Adding element type (line 325)
        float_355095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355095)
        # Adding element type (line 325)
        float_355096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355096)
        # Adding element type (line 325)
        float_355097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355097)
        # Adding element type (line 325)
        float_355098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 24), list_355091, float_355098)
        
        # Processing the call keyword arguments (line 324)
        kwargs_355099 = {}
        # Getting the type of 'assert_allclose' (line 324)
        assert_allclose_355084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 324)
        assert_allclose_call_result_355100 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), assert_allclose_355084, *[kaiser_call_result_355090, list_355091], **kwargs_355099)
        
        
        # Call to assert_allclose(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Call to kaiser(...): (line 328)
        # Processing the call arguments (line 328)
        int_355104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 38), 'int')
        float_355105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 41), 'float')
        # Processing the call keyword arguments (line 328)
        kwargs_355106 = {}
        # Getting the type of 'signal' (line 328)
        signal_355102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 328)
        kaiser_355103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 24), signal_355102, 'kaiser')
        # Calling kaiser(args, kwargs) (line 328)
        kaiser_call_result_355107 = invoke(stypy.reporting.localization.Localization(__file__, 328, 24), kaiser_355103, *[int_355104, float_355105], **kwargs_355106)
        
        
        # Obtaining an instance of the builtin type 'list' (line 329)
        list_355108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 329)
        # Adding element type (line 329)
        float_355109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355109)
        # Adding element type (line 329)
        float_355110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355110)
        # Adding element type (line 329)
        float_355111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355111)
        # Adding element type (line 329)
        float_355112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355112)
        # Adding element type (line 329)
        float_355113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355113)
        # Adding element type (line 329)
        float_355114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 24), list_355108, float_355114)
        
        # Processing the call keyword arguments (line 328)
        kwargs_355115 = {}
        # Getting the type of 'assert_allclose' (line 328)
        assert_allclose_355101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 328)
        assert_allclose_call_result_355116 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), assert_allclose_355101, *[kaiser_call_result_355107, list_355108], **kwargs_355115)
        
        
        # Call to assert_allclose(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to kaiser(...): (line 332)
        # Processing the call arguments (line 332)
        int_355120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 38), 'int')
        float_355121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 41), 'float')
        # Processing the call keyword arguments (line 332)
        kwargs_355122 = {}
        # Getting the type of 'signal' (line 332)
        signal_355118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 332)
        kaiser_355119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 24), signal_355118, 'kaiser')
        # Calling kaiser(args, kwargs) (line 332)
        kaiser_call_result_355123 = invoke(stypy.reporting.localization.Localization(__file__, 332, 24), kaiser_355119, *[int_355120, float_355121], **kwargs_355122)
        
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_355124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        float_355125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355125)
        # Adding element type (line 333)
        float_355126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355126)
        # Adding element type (line 333)
        float_355127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355127)
        # Adding element type (line 333)
        float_355128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355128)
        # Adding element type (line 333)
        float_355129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355129)
        # Adding element type (line 333)
        float_355130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355130)
        # Adding element type (line 333)
        float_355131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 24), list_355124, float_355131)
        
        # Processing the call keyword arguments (line 332)
        kwargs_355132 = {}
        # Getting the type of 'assert_allclose' (line 332)
        assert_allclose_355117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 332)
        assert_allclose_call_result_355133 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), assert_allclose_355117, *[kaiser_call_result_355123, list_355124], **kwargs_355132)
        
        
        # Call to assert_allclose(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Call to kaiser(...): (line 336)
        # Processing the call arguments (line 336)
        int_355137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 38), 'int')
        float_355138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 41), 'float')
        # Getting the type of 'False' (line 336)
        False_355139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 46), 'False', False)
        # Processing the call keyword arguments (line 336)
        kwargs_355140 = {}
        # Getting the type of 'signal' (line 336)
        signal_355135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 336)
        kaiser_355136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), signal_355135, 'kaiser')
        # Calling kaiser(args, kwargs) (line 336)
        kaiser_call_result_355141 = invoke(stypy.reporting.localization.Localization(__file__, 336, 24), kaiser_355136, *[int_355137, float_355138, False_355139], **kwargs_355140)
        
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_355142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        float_355143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355143)
        # Adding element type (line 337)
        float_355144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355144)
        # Adding element type (line 337)
        float_355145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355145)
        # Adding element type (line 337)
        float_355146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355146)
        # Adding element type (line 337)
        float_355147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355147)
        # Adding element type (line 337)
        float_355148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 24), list_355142, float_355148)
        
        # Processing the call keyword arguments (line 336)
        kwargs_355149 = {}
        # Getting the type of 'assert_allclose' (line 336)
        assert_allclose_355134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 336)
        assert_allclose_call_result_355150 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), assert_allclose_355134, *[kaiser_call_result_355141, list_355142], **kwargs_355149)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_355151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355151


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 317, 0, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKaiser.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKaiser' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'TestKaiser', TestKaiser)
# Declaration of the 'TestNuttall' class

class TestNuttall(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 344, 4, False)
        # Assigning a type to the variable 'self' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestNuttall.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_function_name', 'TestNuttall.test_basic')
        TestNuttall.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestNuttall.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestNuttall.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNuttall.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Call to nuttall(...): (line 345)
        # Processing the call arguments (line 345)
        int_355155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 39), 'int')
        # Processing the call keyword arguments (line 345)
        # Getting the type of 'False' (line 345)
        False_355156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 46), 'False', False)
        keyword_355157 = False_355156
        kwargs_355158 = {'sym': keyword_355157}
        # Getting the type of 'signal' (line 345)
        signal_355153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'signal', False)
        # Obtaining the member 'nuttall' of a type (line 345)
        nuttall_355154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 24), signal_355153, 'nuttall')
        # Calling nuttall(args, kwargs) (line 345)
        nuttall_call_result_355159 = invoke(stypy.reporting.localization.Localization(__file__, 345, 24), nuttall_355154, *[int_355155], **kwargs_355158)
        
        
        # Obtaining an instance of the builtin type 'list' (line 346)
        list_355160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 346)
        # Adding element type (line 346)
        float_355161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355161)
        # Adding element type (line 346)
        float_355162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355162)
        # Adding element type (line 346)
        float_355163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355163)
        # Adding element type (line 346)
        float_355164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355164)
        # Adding element type (line 346)
        float_355165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355165)
        # Adding element type (line 346)
        float_355166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_355160, float_355166)
        
        # Processing the call keyword arguments (line 345)
        kwargs_355167 = {}
        # Getting the type of 'assert_allclose' (line 345)
        assert_allclose_355152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 345)
        assert_allclose_call_result_355168 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_allclose_355152, *[nuttall_call_result_355159, list_355160], **kwargs_355167)
        
        
        # Call to assert_allclose(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Call to nuttall(...): (line 348)
        # Processing the call arguments (line 348)
        int_355172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 39), 'int')
        # Processing the call keyword arguments (line 348)
        # Getting the type of 'False' (line 348)
        False_355173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 46), 'False', False)
        keyword_355174 = False_355173
        kwargs_355175 = {'sym': keyword_355174}
        # Getting the type of 'signal' (line 348)
        signal_355170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 24), 'signal', False)
        # Obtaining the member 'nuttall' of a type (line 348)
        nuttall_355171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 24), signal_355170, 'nuttall')
        # Calling nuttall(args, kwargs) (line 348)
        nuttall_call_result_355176 = invoke(stypy.reporting.localization.Localization(__file__, 348, 24), nuttall_355171, *[int_355172], **kwargs_355175)
        
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_355177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        float_355178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355178)
        # Adding element type (line 349)
        float_355179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355179)
        # Adding element type (line 349)
        float_355180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355180)
        # Adding element type (line 349)
        float_355181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355181)
        # Adding element type (line 349)
        float_355182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355182)
        # Adding element type (line 349)
        float_355183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355183)
        # Adding element type (line 349)
        float_355184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 24), list_355177, float_355184)
        
        # Processing the call keyword arguments (line 348)
        kwargs_355185 = {}
        # Getting the type of 'assert_allclose' (line 348)
        assert_allclose_355169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 348)
        assert_allclose_call_result_355186 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), assert_allclose_355169, *[nuttall_call_result_355176, list_355177], **kwargs_355185)
        
        
        # Call to assert_allclose(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Call to nuttall(...): (line 352)
        # Processing the call arguments (line 352)
        int_355190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 39), 'int')
        # Processing the call keyword arguments (line 352)
        kwargs_355191 = {}
        # Getting the type of 'signal' (line 352)
        signal_355188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 24), 'signal', False)
        # Obtaining the member 'nuttall' of a type (line 352)
        nuttall_355189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 24), signal_355188, 'nuttall')
        # Calling nuttall(args, kwargs) (line 352)
        nuttall_call_result_355192 = invoke(stypy.reporting.localization.Localization(__file__, 352, 24), nuttall_355189, *[int_355190], **kwargs_355191)
        
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_355193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        # Adding element type (line 353)
        float_355194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355194)
        # Adding element type (line 353)
        float_355195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355195)
        # Adding element type (line 353)
        float_355196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355196)
        # Adding element type (line 353)
        float_355197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355197)
        # Adding element type (line 353)
        float_355198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355198)
        # Adding element type (line 353)
        float_355199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 24), list_355193, float_355199)
        
        # Processing the call keyword arguments (line 352)
        kwargs_355200 = {}
        # Getting the type of 'assert_allclose' (line 352)
        assert_allclose_355187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 352)
        assert_allclose_call_result_355201 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), assert_allclose_355187, *[nuttall_call_result_355192, list_355193], **kwargs_355200)
        
        
        # Call to assert_allclose(...): (line 355)
        # Processing the call arguments (line 355)
        
        # Call to nuttall(...): (line 355)
        # Processing the call arguments (line 355)
        int_355205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 39), 'int')
        # Getting the type of 'True' (line 355)
        True_355206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 42), 'True', False)
        # Processing the call keyword arguments (line 355)
        kwargs_355207 = {}
        # Getting the type of 'signal' (line 355)
        signal_355203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'signal', False)
        # Obtaining the member 'nuttall' of a type (line 355)
        nuttall_355204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), signal_355203, 'nuttall')
        # Calling nuttall(args, kwargs) (line 355)
        nuttall_call_result_355208 = invoke(stypy.reporting.localization.Localization(__file__, 355, 24), nuttall_355204, *[int_355205, True_355206], **kwargs_355207)
        
        
        # Obtaining an instance of the builtin type 'list' (line 356)
        list_355209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 356)
        # Adding element type (line 356)
        float_355210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355210)
        # Adding element type (line 356)
        float_355211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355211)
        # Adding element type (line 356)
        float_355212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355212)
        # Adding element type (line 356)
        float_355213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355213)
        # Adding element type (line 356)
        float_355214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355214)
        # Adding element type (line 356)
        float_355215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355215)
        # Adding element type (line 356)
        float_355216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 24), list_355209, float_355216)
        
        # Processing the call keyword arguments (line 355)
        kwargs_355217 = {}
        # Getting the type of 'assert_allclose' (line 355)
        assert_allclose_355202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 355)
        assert_allclose_call_result_355218 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), assert_allclose_355202, *[nuttall_call_result_355208, list_355209], **kwargs_355217)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 344)
        stypy_return_type_355219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355219


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 342, 0, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestNuttall.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestNuttall' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'TestNuttall', TestNuttall)
# Declaration of the 'TestParzen' class

class TestParzen(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 362, 4, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestParzen.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestParzen.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestParzen.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestParzen.test_basic.__dict__.__setitem__('stypy_function_name', 'TestParzen.test_basic')
        TestParzen.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestParzen.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestParzen.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestParzen.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestParzen.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestParzen.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestParzen.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestParzen.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Call to parzen(...): (line 363)
        # Processing the call arguments (line 363)
        int_355223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'int')
        # Processing the call keyword arguments (line 363)
        kwargs_355224 = {}
        # Getting the type of 'signal' (line 363)
        signal_355221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'signal', False)
        # Obtaining the member 'parzen' of a type (line 363)
        parzen_355222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), signal_355221, 'parzen')
        # Calling parzen(args, kwargs) (line 363)
        parzen_call_result_355225 = invoke(stypy.reporting.localization.Localization(__file__, 363, 24), parzen_355222, *[int_355223], **kwargs_355224)
        
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_355226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        float_355227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355227)
        # Adding element type (line 364)
        float_355228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355228)
        # Adding element type (line 364)
        float_355229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355229)
        # Adding element type (line 364)
        float_355230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355230)
        # Adding element type (line 364)
        float_355231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355231)
        # Adding element type (line 364)
        float_355232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 24), list_355226, float_355232)
        
        # Processing the call keyword arguments (line 363)
        kwargs_355233 = {}
        # Getting the type of 'assert_allclose' (line 363)
        assert_allclose_355220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 363)
        assert_allclose_call_result_355234 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assert_allclose_355220, *[parzen_call_result_355225, list_355226], **kwargs_355233)
        
        
        # Call to assert_allclose(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Call to parzen(...): (line 366)
        # Processing the call arguments (line 366)
        int_355238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 38), 'int')
        # Processing the call keyword arguments (line 366)
        # Getting the type of 'True' (line 366)
        True_355239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 45), 'True', False)
        keyword_355240 = True_355239
        kwargs_355241 = {'sym': keyword_355240}
        # Getting the type of 'signal' (line 366)
        signal_355236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 24), 'signal', False)
        # Obtaining the member 'parzen' of a type (line 366)
        parzen_355237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 24), signal_355236, 'parzen')
        # Calling parzen(args, kwargs) (line 366)
        parzen_call_result_355242 = invoke(stypy.reporting.localization.Localization(__file__, 366, 24), parzen_355237, *[int_355238], **kwargs_355241)
        
        
        # Obtaining an instance of the builtin type 'list' (line 367)
        list_355243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 367)
        # Adding element type (line 367)
        float_355244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355244)
        # Adding element type (line 367)
        float_355245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355245)
        # Adding element type (line 367)
        float_355246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355246)
        # Adding element type (line 367)
        float_355247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355247)
        # Adding element type (line 367)
        float_355248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355248)
        # Adding element type (line 367)
        float_355249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355249)
        # Adding element type (line 367)
        float_355250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 24), list_355243, float_355250)
        
        # Processing the call keyword arguments (line 366)
        kwargs_355251 = {}
        # Getting the type of 'assert_allclose' (line 366)
        assert_allclose_355235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 366)
        assert_allclose_call_result_355252 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), assert_allclose_355235, *[parzen_call_result_355242, list_355243], **kwargs_355251)
        
        
        # Call to assert_allclose(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Call to parzen(...): (line 370)
        # Processing the call arguments (line 370)
        int_355256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 38), 'int')
        # Getting the type of 'False' (line 370)
        False_355257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 41), 'False', False)
        # Processing the call keyword arguments (line 370)
        kwargs_355258 = {}
        # Getting the type of 'signal' (line 370)
        signal_355254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'signal', False)
        # Obtaining the member 'parzen' of a type (line 370)
        parzen_355255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), signal_355254, 'parzen')
        # Calling parzen(args, kwargs) (line 370)
        parzen_call_result_355259 = invoke(stypy.reporting.localization.Localization(__file__, 370, 24), parzen_355255, *[int_355256, False_355257], **kwargs_355258)
        
        
        # Obtaining an instance of the builtin type 'list' (line 371)
        list_355260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 371)
        # Adding element type (line 371)
        float_355261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355261)
        # Adding element type (line 371)
        float_355262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355262)
        # Adding element type (line 371)
        float_355263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355263)
        # Adding element type (line 371)
        float_355264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355264)
        # Adding element type (line 371)
        float_355265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355265)
        # Adding element type (line 371)
        float_355266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 24), list_355260, float_355266)
        
        # Processing the call keyword arguments (line 370)
        kwargs_355267 = {}
        # Getting the type of 'assert_allclose' (line 370)
        assert_allclose_355253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 370)
        assert_allclose_call_result_355268 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert_allclose_355253, *[parzen_call_result_355259, list_355260], **kwargs_355267)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_355269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355269


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 360, 0, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestParzen.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestParzen' (line 360)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 0), 'TestParzen', TestParzen)
# Declaration of the 'TestTriang' class

class TestTriang(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTriang.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTriang.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTriang.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTriang.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTriang.test_basic')
        TestTriang.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTriang.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTriang.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTriang.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTriang.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTriang.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTriang.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTriang.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to assert_allclose(...): (line 380)
        # Processing the call arguments (line 380)
        
        # Call to triang(...): (line 380)
        # Processing the call arguments (line 380)
        int_355273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 38), 'int')
        # Getting the type of 'True' (line 380)
        True_355274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 41), 'True', False)
        # Processing the call keyword arguments (line 380)
        kwargs_355275 = {}
        # Getting the type of 'signal' (line 380)
        signal_355271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'signal', False)
        # Obtaining the member 'triang' of a type (line 380)
        triang_355272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 24), signal_355271, 'triang')
        # Calling triang(args, kwargs) (line 380)
        triang_call_result_355276 = invoke(stypy.reporting.localization.Localization(__file__, 380, 24), triang_355272, *[int_355273, True_355274], **kwargs_355275)
        
        
        # Obtaining an instance of the builtin type 'list' (line 381)
        list_355277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 381)
        # Adding element type (line 381)
        int_355278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 25), 'int')
        int_355279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 27), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355280 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 25), 'div', int_355278, int_355279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355280)
        # Adding element type (line 381)
        int_355281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 30), 'int')
        int_355282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 32), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355283 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 30), 'div', int_355281, int_355282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355283)
        # Adding element type (line 381)
        int_355284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 35), 'int')
        int_355285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 37), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355286 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 35), 'div', int_355284, int_355285)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355286)
        # Adding element type (line 381)
        int_355287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 40), 'int')
        int_355288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 42), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355289 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 40), 'div', int_355287, int_355288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355289)
        # Adding element type (line 381)
        int_355290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 45), 'int')
        int_355291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 47), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355292 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 45), 'div', int_355290, int_355291)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355292)
        # Adding element type (line 381)
        int_355293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 50), 'int')
        int_355294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 52), 'int')
        # Applying the binary operator 'div' (line 381)
        result_div_355295 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 50), 'div', int_355293, int_355294)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 24), list_355277, result_div_355295)
        
        # Processing the call keyword arguments (line 380)
        kwargs_355296 = {}
        # Getting the type of 'assert_allclose' (line 380)
        assert_allclose_355270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 380)
        assert_allclose_call_result_355297 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), assert_allclose_355270, *[triang_call_result_355276, list_355277], **kwargs_355296)
        
        
        # Call to assert_allclose(...): (line 382)
        # Processing the call arguments (line 382)
        
        # Call to triang(...): (line 382)
        # Processing the call arguments (line 382)
        int_355301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 38), 'int')
        # Processing the call keyword arguments (line 382)
        kwargs_355302 = {}
        # Getting the type of 'signal' (line 382)
        signal_355299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 24), 'signal', False)
        # Obtaining the member 'triang' of a type (line 382)
        triang_355300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 24), signal_355299, 'triang')
        # Calling triang(args, kwargs) (line 382)
        triang_call_result_355303 = invoke(stypy.reporting.localization.Localization(__file__, 382, 24), triang_355300, *[int_355301], **kwargs_355302)
        
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_355304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        # Adding element type (line 383)
        int_355305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 25), 'int')
        int_355306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 27), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355307 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 25), 'div', int_355305, int_355306)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355307)
        # Adding element type (line 383)
        int_355308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 30), 'int')
        int_355309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 32), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355310 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 30), 'div', int_355308, int_355309)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355310)
        # Adding element type (line 383)
        int_355311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 35), 'int')
        int_355312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 37), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355313 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 35), 'div', int_355311, int_355312)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355313)
        # Adding element type (line 383)
        int_355314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, int_355314)
        # Adding element type (line 383)
        int_355315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 43), 'int')
        int_355316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 45), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355317 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 43), 'div', int_355315, int_355316)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355317)
        # Adding element type (line 383)
        int_355318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 48), 'int')
        int_355319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 50), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355320 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 48), 'div', int_355318, int_355319)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355320)
        # Adding element type (line 383)
        int_355321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 53), 'int')
        int_355322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 55), 'int')
        # Applying the binary operator 'div' (line 383)
        result_div_355323 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 53), 'div', int_355321, int_355322)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 24), list_355304, result_div_355323)
        
        # Processing the call keyword arguments (line 382)
        kwargs_355324 = {}
        # Getting the type of 'assert_allclose' (line 382)
        assert_allclose_355298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 382)
        assert_allclose_call_result_355325 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), assert_allclose_355298, *[triang_call_result_355303, list_355304], **kwargs_355324)
        
        
        # Call to assert_allclose(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Call to triang(...): (line 384)
        # Processing the call arguments (line 384)
        int_355329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 38), 'int')
        # Processing the call keyword arguments (line 384)
        # Getting the type of 'False' (line 384)
        False_355330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 45), 'False', False)
        keyword_355331 = False_355330
        kwargs_355332 = {'sym': keyword_355331}
        # Getting the type of 'signal' (line 384)
        signal_355327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 24), 'signal', False)
        # Obtaining the member 'triang' of a type (line 384)
        triang_355328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 24), signal_355327, 'triang')
        # Calling triang(args, kwargs) (line 384)
        triang_call_result_355333 = invoke(stypy.reporting.localization.Localization(__file__, 384, 24), triang_355328, *[int_355329], **kwargs_355332)
        
        
        # Obtaining an instance of the builtin type 'list' (line 385)
        list_355334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 385)
        # Adding element type (line 385)
        int_355335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 25), 'int')
        int_355336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 27), 'int')
        # Applying the binary operator 'div' (line 385)
        result_div_355337 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 25), 'div', int_355335, int_355336)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, result_div_355337)
        # Adding element type (line 385)
        int_355338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'int')
        int_355339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 32), 'int')
        # Applying the binary operator 'div' (line 385)
        result_div_355340 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 30), 'div', int_355338, int_355339)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, result_div_355340)
        # Adding element type (line 385)
        int_355341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 35), 'int')
        int_355342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 37), 'int')
        # Applying the binary operator 'div' (line 385)
        result_div_355343 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 35), 'div', int_355341, int_355342)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, result_div_355343)
        # Adding element type (line 385)
        int_355344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, int_355344)
        # Adding element type (line 385)
        int_355345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 43), 'int')
        int_355346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 45), 'int')
        # Applying the binary operator 'div' (line 385)
        result_div_355347 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 43), 'div', int_355345, int_355346)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, result_div_355347)
        # Adding element type (line 385)
        int_355348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 48), 'int')
        int_355349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 50), 'int')
        # Applying the binary operator 'div' (line 385)
        result_div_355350 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 48), 'div', int_355348, int_355349)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 24), list_355334, result_div_355350)
        
        # Processing the call keyword arguments (line 384)
        kwargs_355351 = {}
        # Getting the type of 'assert_allclose' (line 384)
        assert_allclose_355326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 384)
        assert_allclose_call_result_355352 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), assert_allclose_355326, *[triang_call_result_355333, list_355334], **kwargs_355351)
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_355353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355353


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTriang.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTriang' (line 376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'TestTriang', TestTriang)

# Assigning a Dict to a Name (line 388):

# Obtaining an instance of the builtin type 'dict' (line 388)
dict_355354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 388)
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 389)
tuple_355355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 389)
# Adding element type (line 389)
int_355356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 5), tuple_355355, int_355356)
# Adding element type (line 389)
float_355357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 5), tuple_355355, float_355357)
# Adding element type (line 389)
# Getting the type of 'True' (line 389)
True_355358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 5), tuple_355355, True_355358)


# Call to array(...): (line 389)
# Processing the call arguments (line 389)

# Obtaining an instance of the builtin type 'list' (line 389)
list_355360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 389)
# Adding element type (line 389)
float_355361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 26), list_355360, float_355361)
# Adding element type (line 389)
float_355362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 26), list_355360, float_355362)
# Adding element type (line 389)
float_355363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 26), list_355360, float_355363)
# Adding element type (line 389)
float_355364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 26), list_355360, float_355364)

# Processing the call keyword arguments (line 389)
kwargs_355365 = {}
# Getting the type of 'array' (line 389)
array_355359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'array', False)
# Calling array(args, kwargs) (line 389)
array_call_result_355366 = invoke(stypy.reporting.localization.Localization(__file__, 389, 20), array_355359, *[list_355360], **kwargs_355365)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355355, array_call_result_355366))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 390)
tuple_355367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 390)
# Adding element type (line 390)
int_355368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 5), tuple_355367, int_355368)
# Adding element type (line 390)
float_355369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 5), tuple_355367, float_355369)
# Adding element type (line 390)
# Getting the type of 'True' (line 390)
True_355370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 5), tuple_355367, True_355370)


# Call to array(...): (line 390)
# Processing the call arguments (line 390)

# Obtaining an instance of the builtin type 'list' (line 390)
list_355372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 390)
# Adding element type (line 390)
float_355373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 26), list_355372, float_355373)
# Adding element type (line 390)
float_355374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 26), list_355372, float_355374)
# Adding element type (line 390)
float_355375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 26), list_355372, float_355375)
# Adding element type (line 390)
float_355376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 26), list_355372, float_355376)

# Processing the call keyword arguments (line 390)
kwargs_355377 = {}
# Getting the type of 'array' (line 390)
array_355371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 20), 'array', False)
# Calling array(args, kwargs) (line 390)
array_call_result_355378 = invoke(stypy.reporting.localization.Localization(__file__, 390, 20), array_355371, *[list_355372], **kwargs_355377)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355367, array_call_result_355378))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 392)
tuple_355379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 392)
# Adding element type (line 392)
int_355380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 5), tuple_355379, int_355380)
# Adding element type (line 392)
float_355381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 5), tuple_355379, float_355381)
# Adding element type (line 392)
# Getting the type of 'True' (line 392)
True_355382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 5), tuple_355379, True_355382)


# Call to array(...): (line 392)
# Processing the call arguments (line 392)

# Obtaining an instance of the builtin type 'list' (line 392)
list_355384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 392)
# Adding element type (line 392)
float_355385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 26), list_355384, float_355385)
# Adding element type (line 392)
float_355386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 26), list_355384, float_355386)
# Adding element type (line 392)
float_355387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 26), list_355384, float_355387)
# Adding element type (line 392)
float_355388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 26), list_355384, float_355388)

# Processing the call keyword arguments (line 392)
kwargs_355389 = {}
# Getting the type of 'array' (line 392)
array_355383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'array', False)
# Calling array(args, kwargs) (line 392)
array_call_result_355390 = invoke(stypy.reporting.localization.Localization(__file__, 392, 20), array_355383, *[list_355384], **kwargs_355389)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355379, array_call_result_355390))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 393)
tuple_355391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 393)
# Adding element type (line 393)
int_355392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 5), tuple_355391, int_355392)
# Adding element type (line 393)
float_355393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 5), tuple_355391, float_355393)
# Adding element type (line 393)
# Getting the type of 'False' (line 393)
False_355394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 5), tuple_355391, False_355394)


# Call to array(...): (line 393)
# Processing the call arguments (line 393)

# Obtaining an instance of the builtin type 'list' (line 393)
list_355396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 393)
# Adding element type (line 393)
float_355397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 27), list_355396, float_355397)
# Adding element type (line 393)
float_355398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 27), list_355396, float_355398)
# Adding element type (line 393)
float_355399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 27), list_355396, float_355399)
# Adding element type (line 393)
float_355400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 27), list_355396, float_355400)

# Processing the call keyword arguments (line 393)
kwargs_355401 = {}
# Getting the type of 'array' (line 393)
array_355395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 21), 'array', False)
# Calling array(args, kwargs) (line 393)
array_call_result_355402 = invoke(stypy.reporting.localization.Localization(__file__, 393, 21), array_355395, *[list_355396], **kwargs_355401)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355391, array_call_result_355402))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 394)
tuple_355403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 394)
# Adding element type (line 394)
int_355404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 5), tuple_355403, int_355404)
# Adding element type (line 394)
float_355405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 5), tuple_355403, float_355405)
# Adding element type (line 394)
# Getting the type of 'False' (line 394)
False_355406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 13), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 5), tuple_355403, False_355406)


# Call to array(...): (line 394)
# Processing the call arguments (line 394)

# Obtaining an instance of the builtin type 'list' (line 394)
list_355408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 394)
# Adding element type (line 394)
float_355409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), list_355408, float_355409)
# Adding element type (line 394)
float_355410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), list_355408, float_355410)
# Adding element type (line 394)
float_355411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), list_355408, float_355411)
# Adding element type (line 394)
float_355412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 27), list_355408, float_355412)

# Processing the call keyword arguments (line 394)
kwargs_355413 = {}
# Getting the type of 'array' (line 394)
array_355407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'array', False)
# Calling array(args, kwargs) (line 394)
array_call_result_355414 = invoke(stypy.reporting.localization.Localization(__file__, 394, 21), array_355407, *[list_355408], **kwargs_355413)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355403, array_call_result_355414))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 396)
tuple_355415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 396)
# Adding element type (line 396)
int_355416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 5), tuple_355415, int_355416)
# Adding element type (line 396)
float_355417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 5), tuple_355415, float_355417)
# Adding element type (line 396)
# Getting the type of 'False' (line 396)
False_355418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 13), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 5), tuple_355415, False_355418)


# Call to array(...): (line 396)
# Processing the call arguments (line 396)

# Obtaining an instance of the builtin type 'list' (line 396)
list_355420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 396)
# Adding element type (line 396)
float_355421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), list_355420, float_355421)
# Adding element type (line 396)
float_355422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), list_355420, float_355422)
# Adding element type (line 396)
float_355423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), list_355420, float_355423)
# Adding element type (line 396)
float_355424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 27), list_355420, float_355424)

# Processing the call keyword arguments (line 396)
kwargs_355425 = {}
# Getting the type of 'array' (line 396)
array_355419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 21), 'array', False)
# Calling array(args, kwargs) (line 396)
array_call_result_355426 = invoke(stypy.reporting.localization.Localization(__file__, 396, 21), array_355419, *[list_355420], **kwargs_355425)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355415, array_call_result_355426))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 397)
tuple_355427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 397)
# Adding element type (line 397)
int_355428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 5), tuple_355427, int_355428)
# Adding element type (line 397)
float_355429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 5), tuple_355427, float_355429)
# Adding element type (line 397)
# Getting the type of 'True' (line 397)
True_355430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 5), tuple_355427, True_355430)


# Call to array(...): (line 397)
# Processing the call arguments (line 397)

# Obtaining an instance of the builtin type 'list' (line 397)
list_355432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 397)
# Adding element type (line 397)
float_355433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 26), list_355432, float_355433)
# Adding element type (line 397)
float_355434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 26), list_355432, float_355434)
# Adding element type (line 397)
float_355435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 26), list_355432, float_355435)
# Adding element type (line 397)
float_355436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 26), list_355432, float_355436)
# Adding element type (line 397)
float_355437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 26), list_355432, float_355437)

# Processing the call keyword arguments (line 397)
kwargs_355438 = {}
# Getting the type of 'array' (line 397)
array_355431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 20), 'array', False)
# Calling array(args, kwargs) (line 397)
array_call_result_355439 = invoke(stypy.reporting.localization.Localization(__file__, 397, 20), array_355431, *[list_355432], **kwargs_355438)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355427, array_call_result_355439))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 398)
tuple_355440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 398)
# Adding element type (line 398)
int_355441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 5), tuple_355440, int_355441)
# Adding element type (line 398)
float_355442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 5), tuple_355440, float_355442)
# Adding element type (line 398)
# Getting the type of 'True' (line 398)
True_355443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 5), tuple_355440, True_355443)


# Call to array(...): (line 398)
# Processing the call arguments (line 398)

# Obtaining an instance of the builtin type 'list' (line 398)
list_355445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 398)
# Adding element type (line 398)
float_355446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_355445, float_355446)
# Adding element type (line 398)
float_355447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_355445, float_355447)
# Adding element type (line 398)
float_355448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_355445, float_355448)
# Adding element type (line 398)
float_355449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_355445, float_355449)
# Adding element type (line 398)
float_355450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 53), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 26), list_355445, float_355450)

# Processing the call keyword arguments (line 398)
kwargs_355451 = {}
# Getting the type of 'array' (line 398)
array_355444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'array', False)
# Calling array(args, kwargs) (line 398)
array_call_result_355452 = invoke(stypy.reporting.localization.Localization(__file__, 398, 20), array_355444, *[list_355445], **kwargs_355451)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355440, array_call_result_355452))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 400)
tuple_355453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 400)
# Adding element type (line 400)
int_355454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 5), tuple_355453, int_355454)
# Adding element type (line 400)
float_355455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 5), tuple_355453, float_355455)
# Adding element type (line 400)
# Getting the type of 'True' (line 400)
True_355456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 5), tuple_355453, True_355456)


# Call to array(...): (line 400)
# Processing the call arguments (line 400)

# Obtaining an instance of the builtin type 'list' (line 400)
list_355458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 400)
# Adding element type (line 400)
float_355459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 27), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 26), list_355458, float_355459)
# Adding element type (line 400)
float_355460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 26), list_355458, float_355460)
# Adding element type (line 400)
float_355461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 26), list_355458, float_355461)
# Adding element type (line 400)
float_355462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 26), list_355458, float_355462)
# Adding element type (line 400)
float_355463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 26), list_355458, float_355463)

# Processing the call keyword arguments (line 400)
kwargs_355464 = {}
# Getting the type of 'array' (line 400)
array_355457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 20), 'array', False)
# Calling array(args, kwargs) (line 400)
array_call_result_355465 = invoke(stypy.reporting.localization.Localization(__file__, 400, 20), array_355457, *[list_355458], **kwargs_355464)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355453, array_call_result_355465))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 402)
tuple_355466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 402)
# Adding element type (line 402)
int_355467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 5), tuple_355466, int_355467)
# Adding element type (line 402)
int_355468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 5), tuple_355466, int_355468)


# Obtaining an instance of the builtin type 'list' (line 402)
list_355469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 402)
# Adding element type (line 402)
int_355470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355470)
# Adding element type (line 402)
int_355471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355471)
# Adding element type (line 402)
int_355472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355472)
# Adding element type (line 402)
int_355473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355473)
# Adding element type (line 402)
int_355474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355474)
# Adding element type (line 402)
int_355475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 402, 12), list_355469, int_355475)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355466, list_355469))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 403)
tuple_355476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 403)
# Adding element type (line 403)
int_355477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 5), tuple_355476, int_355477)
# Adding element type (line 403)
int_355478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 5), tuple_355476, int_355478)


# Obtaining an instance of the builtin type 'list' (line 403)
list_355479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 403)
# Adding element type (line 403)
int_355480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355480)
# Adding element type (line 403)
int_355481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355481)
# Adding element type (line 403)
int_355482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355482)
# Adding element type (line 403)
int_355483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355483)
# Adding element type (line 403)
int_355484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355484)
# Adding element type (line 403)
int_355485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355485)
# Adding element type (line 403)
int_355486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_355479, int_355486)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355476, list_355479))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 404)
tuple_355487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 404)
# Adding element type (line 404)
int_355488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 5), tuple_355487, int_355488)
# Adding element type (line 404)
float_355489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 5), tuple_355487, float_355489)


# Obtaining an instance of the builtin type 'list' (line 404)
list_355490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 404)
# Adding element type (line 404)
int_355491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355491)
# Adding element type (line 404)
int_355492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355492)
# Adding element type (line 404)
int_355493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355493)
# Adding element type (line 404)
int_355494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355494)
# Adding element type (line 404)
int_355495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355495)
# Adding element type (line 404)
int_355496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 14), list_355490, int_355496)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355487, list_355490))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 405)
tuple_355497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 405)
# Adding element type (line 405)
int_355498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 5), tuple_355497, int_355498)
# Adding element type (line 405)
float_355499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 5), tuple_355497, float_355499)


# Obtaining an instance of the builtin type 'list' (line 405)
list_355500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 405)
# Adding element type (line 405)
int_355501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355501)
# Adding element type (line 405)
int_355502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355502)
# Adding element type (line 405)
int_355503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355503)
# Adding element type (line 405)
int_355504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355504)
# Adding element type (line 405)
int_355505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355505)
# Adding element type (line 405)
int_355506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355506)
# Adding element type (line 405)
int_355507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 14), list_355500, int_355507)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355497, list_355500))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 406)
tuple_355508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 406)
# Adding element type (line 406)
int_355509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 5), tuple_355508, int_355509)


# Obtaining an instance of the builtin type 'list' (line 406)
list_355510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 406)
# Adding element type (line 406)
int_355511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, int_355511)
# Adding element type (line 406)
float_355512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, float_355512)
# Adding element type (line 406)
float_355513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, float_355513)
# Adding element type (line 406)
float_355514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, float_355514)
# Adding element type (line 406)
float_355515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, float_355515)
# Adding element type (line 406)
int_355516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 64), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 10), list_355510, int_355516)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355508, list_355510))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 407)
tuple_355517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 407)
# Adding element type (line 407)
int_355518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 5), tuple_355517, int_355518)


# Obtaining an instance of the builtin type 'list' (line 407)
list_355519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 407)
# Adding element type (line 407)
int_355520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, int_355520)
# Adding element type (line 407)
float_355521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, float_355521)
# Adding element type (line 407)
float_355522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, float_355522)
# Adding element type (line 407)
float_355523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, float_355523)
# Adding element type (line 407)
float_355524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, float_355524)
# Adding element type (line 407)
float_355525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, float_355525)
# Adding element type (line 407)
int_355526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 41), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 10), list_355519, int_355526)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355517, list_355519))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 408)
tuple_355527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 408)
# Adding element type (line 408)
int_355528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 5), tuple_355527, int_355528)
# Adding element type (line 408)
float_355529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 5), tuple_355527, float_355529)


# Obtaining an instance of the builtin type 'list' (line 408)
list_355530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 408)
# Adding element type (line 408)
int_355531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, int_355531)
# Adding element type (line 408)
float_355532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, float_355532)
# Adding element type (line 408)
float_355533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, float_355533)
# Adding element type (line 408)
float_355534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, float_355534)
# Adding element type (line 408)
float_355535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, float_355535)
# Adding element type (line 408)
int_355536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 68), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 14), list_355530, int_355536)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355527, list_355530))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 409)
tuple_355537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 409)
# Adding element type (line 409)
int_355538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 5), tuple_355537, int_355538)
# Adding element type (line 409)
float_355539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 5), tuple_355537, float_355539)


# Obtaining an instance of the builtin type 'list' (line 409)
list_355540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 409)
# Adding element type (line 409)
int_355541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, int_355541)
# Adding element type (line 409)
float_355542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, float_355542)
# Adding element type (line 409)
float_355543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 38), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, float_355543)
# Adding element type (line 409)
float_355544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 58), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, float_355544)
# Adding element type (line 409)
float_355545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, float_355545)
# Adding element type (line 409)
float_355546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, float_355546)
# Adding element type (line 409)
int_355547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 55), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 14), list_355540, int_355547)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355537, list_355540))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 411)
tuple_355548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 411)
# Adding element type (line 411)
int_355549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 5), tuple_355548, int_355549)
# Adding element type (line 411)
int_355550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 5), tuple_355548, int_355550)


# Obtaining an instance of the builtin type 'list' (line 411)
list_355551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 411)
# Adding element type (line 411)
int_355552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, int_355552)
# Adding element type (line 411)
float_355553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, float_355553)
# Adding element type (line 411)
float_355554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, float_355554)
# Adding element type (line 411)
float_355555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, float_355555)
# Adding element type (line 411)
float_355556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 13), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, float_355556)
# Adding element type (line 411)
int_355557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 12), list_355551, int_355557)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355548, list_355551))
# Adding element type (key, value) (line 388)

# Obtaining an instance of the builtin type 'tuple' (line 413)
tuple_355558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 413)
# Adding element type (line 413)
int_355559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 5), tuple_355558, int_355559)
# Adding element type (line 413)
int_355560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 5), tuple_355558, int_355560)


# Obtaining an instance of the builtin type 'list' (line 413)
list_355561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 413)
# Adding element type (line 413)
int_355562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, int_355562)
# Adding element type (line 413)
float_355563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, float_355563)
# Adding element type (line 413)
float_355564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, float_355564)
# Adding element type (line 413)
float_355565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, float_355565)
# Adding element type (line 413)
float_355566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 33), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, float_355566)
# Adding element type (line 413)
float_355567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, float_355567)
# Adding element type (line 413)
int_355568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 12), list_355561, int_355568)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 13), dict_355354, (tuple_355558, list_355561))

# Assigning a type to the variable 'tukey_data' (line 388)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 0), 'tukey_data', dict_355354)
# Declaration of the 'TestTukey' class

class TestTukey(object, ):

    @norecursion
    def test_basic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_basic'
        module_type_store = module_type_store.open_function_context('test_basic', 419, 4, False)
        # Assigning a type to the variable 'self' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTukey.test_basic.__dict__.__setitem__('stypy_localization', localization)
        TestTukey.test_basic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTukey.test_basic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTukey.test_basic.__dict__.__setitem__('stypy_function_name', 'TestTukey.test_basic')
        TestTukey.test_basic.__dict__.__setitem__('stypy_param_names_list', [])
        TestTukey.test_basic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTukey.test_basic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTukey.test_basic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTukey.test_basic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTukey.test_basic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTukey.test_basic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTukey.test_basic', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to items(...): (line 421)
        # Processing the call keyword arguments (line 421)
        kwargs_355571 = {}
        # Getting the type of 'tukey_data' (line 421)
        tukey_data_355569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 20), 'tukey_data', False)
        # Obtaining the member 'items' of a type (line 421)
        items_355570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 20), tukey_data_355569, 'items')
        # Calling items(args, kwargs) (line 421)
        items_call_result_355572 = invoke(stypy.reporting.localization.Localization(__file__, 421, 20), items_355570, *[], **kwargs_355571)
        
        # Testing the type of a for loop iterable (line 421)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 421, 8), items_call_result_355572)
        # Getting the type of the for loop variable (line 421)
        for_loop_var_355573 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 421, 8), items_call_result_355572)
        # Assigning a type to the variable 'k' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), for_loop_var_355573))
        # Assigning a type to the variable 'v' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 421, 8), for_loop_var_355573))
        # SSA begins for a for statement (line 421)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 422)
        # Getting the type of 'v' (line 422)
        v_355574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'v')
        # Getting the type of 'None' (line 422)
        None_355575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'None')
        
        (may_be_355576, more_types_in_union_355577) = may_be_none(v_355574, None_355575)

        if may_be_355576:

            if more_types_in_union_355577:
                # Runtime conditional SSA (line 422)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to assert_raises(...): (line 423)
            # Processing the call arguments (line 423)
            # Getting the type of 'ValueError' (line 423)
            ValueError_355579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 30), 'ValueError', False)
            # Getting the type of 'signal' (line 423)
            signal_355580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'signal', False)
            # Obtaining the member 'tukey' of a type (line 423)
            tukey_355581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 42), signal_355580, 'tukey')
            # Getting the type of 'k' (line 423)
            k_355582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 57), 'k', False)
            # Processing the call keyword arguments (line 423)
            kwargs_355583 = {}
            # Getting the type of 'assert_raises' (line 423)
            assert_raises_355578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 16), 'assert_raises', False)
            # Calling assert_raises(args, kwargs) (line 423)
            assert_raises_call_result_355584 = invoke(stypy.reporting.localization.Localization(__file__, 423, 16), assert_raises_355578, *[ValueError_355579, tukey_355581, k_355582], **kwargs_355583)
            

            if more_types_in_union_355577:
                # Runtime conditional SSA for else branch (line 422)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_355576) or more_types_in_union_355577):
            
            # Assigning a Call to a Name (line 425):
            
            # Call to tukey(...): (line 425)
            # Getting the type of 'k' (line 425)
            k_355587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 36), 'k', False)
            # Processing the call keyword arguments (line 425)
            kwargs_355588 = {}
            # Getting the type of 'signal' (line 425)
            signal_355585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'signal', False)
            # Obtaining the member 'tukey' of a type (line 425)
            tukey_355586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 22), signal_355585, 'tukey')
            # Calling tukey(args, kwargs) (line 425)
            tukey_call_result_355589 = invoke(stypy.reporting.localization.Localization(__file__, 425, 22), tukey_355586, *[k_355587], **kwargs_355588)
            
            # Assigning a type to the variable 'win' (line 425)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 16), 'win', tukey_call_result_355589)
            
            # Call to assert_allclose(...): (line 426)
            # Processing the call arguments (line 426)
            # Getting the type of 'win' (line 426)
            win_355591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 32), 'win', False)
            # Getting the type of 'v' (line 426)
            v_355592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 37), 'v', False)
            # Processing the call keyword arguments (line 426)
            float_355593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 45), 'float')
            keyword_355594 = float_355593
            kwargs_355595 = {'rtol': keyword_355594}
            # Getting the type of 'assert_allclose' (line 426)
            assert_allclose_355590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 426)
            assert_allclose_call_result_355596 = invoke(stypy.reporting.localization.Localization(__file__, 426, 16), assert_allclose_355590, *[win_355591, v_355592], **kwargs_355595)
            

            if (may_be_355576 and more_types_in_union_355577):
                # SSA join for if statement (line 422)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_basic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_basic' in the type store
        # Getting the type of 'stypy_return_type' (line 419)
        stypy_return_type_355597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355597)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_basic'
        return stypy_return_type_355597


    @norecursion
    def test_extremes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_extremes'
        module_type_store = module_type_store.open_function_context('test_extremes', 428, 4, False)
        # Assigning a type to the variable 'self' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTukey.test_extremes.__dict__.__setitem__('stypy_localization', localization)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_function_name', 'TestTukey.test_extremes')
        TestTukey.test_extremes.__dict__.__setitem__('stypy_param_names_list', [])
        TestTukey.test_extremes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTukey.test_extremes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTukey.test_extremes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_extremes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_extremes(...)' code ##################

        
        # Assigning a Call to a Name (line 430):
        
        # Call to tukey(...): (line 430)
        # Processing the call arguments (line 430)
        int_355600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 28), 'int')
        int_355601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 33), 'int')
        # Processing the call keyword arguments (line 430)
        kwargs_355602 = {}
        # Getting the type of 'signal' (line 430)
        signal_355598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'signal', False)
        # Obtaining the member 'tukey' of a type (line 430)
        tukey_355599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 15), signal_355598, 'tukey')
        # Calling tukey(args, kwargs) (line 430)
        tukey_call_result_355603 = invoke(stypy.reporting.localization.Localization(__file__, 430, 15), tukey_355599, *[int_355600, int_355601], **kwargs_355602)
        
        # Assigning a type to the variable 'tuk0' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'tuk0', tukey_call_result_355603)
        
        # Assigning a Call to a Name (line 431):
        
        # Call to boxcar(...): (line 431)
        # Processing the call arguments (line 431)
        int_355606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 29), 'int')
        # Processing the call keyword arguments (line 431)
        kwargs_355607 = {}
        # Getting the type of 'signal' (line 431)
        signal_355604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 15), 'signal', False)
        # Obtaining the member 'boxcar' of a type (line 431)
        boxcar_355605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 15), signal_355604, 'boxcar')
        # Calling boxcar(args, kwargs) (line 431)
        boxcar_call_result_355608 = invoke(stypy.reporting.localization.Localization(__file__, 431, 15), boxcar_355605, *[int_355606], **kwargs_355607)
        
        # Assigning a type to the variable 'box0' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'box0', boxcar_call_result_355608)
        
        # Call to assert_array_almost_equal(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'tuk0' (line 432)
        tuk0_355610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'tuk0', False)
        # Getting the type of 'box0' (line 432)
        box0_355611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 40), 'box0', False)
        # Processing the call keyword arguments (line 432)
        kwargs_355612 = {}
        # Getting the type of 'assert_array_almost_equal' (line 432)
        assert_array_almost_equal_355609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 432)
        assert_array_almost_equal_call_result_355613 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), assert_array_almost_equal_355609, *[tuk0_355610, box0_355611], **kwargs_355612)
        
        
        # Assigning a Call to a Name (line 434):
        
        # Call to tukey(...): (line 434)
        # Processing the call arguments (line 434)
        int_355616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 28), 'int')
        int_355617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 33), 'int')
        # Processing the call keyword arguments (line 434)
        kwargs_355618 = {}
        # Getting the type of 'signal' (line 434)
        signal_355614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 15), 'signal', False)
        # Obtaining the member 'tukey' of a type (line 434)
        tukey_355615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 15), signal_355614, 'tukey')
        # Calling tukey(args, kwargs) (line 434)
        tukey_call_result_355619 = invoke(stypy.reporting.localization.Localization(__file__, 434, 15), tukey_355615, *[int_355616, int_355617], **kwargs_355618)
        
        # Assigning a type to the variable 'tuk1' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'tuk1', tukey_call_result_355619)
        
        # Assigning a Call to a Name (line 435):
        
        # Call to hann(...): (line 435)
        # Processing the call arguments (line 435)
        int_355622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'int')
        # Processing the call keyword arguments (line 435)
        kwargs_355623 = {}
        # Getting the type of 'signal' (line 435)
        signal_355620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'signal', False)
        # Obtaining the member 'hann' of a type (line 435)
        hann_355621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 15), signal_355620, 'hann')
        # Calling hann(args, kwargs) (line 435)
        hann_call_result_355624 = invoke(stypy.reporting.localization.Localization(__file__, 435, 15), hann_355621, *[int_355622], **kwargs_355623)
        
        # Assigning a type to the variable 'han1' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'han1', hann_call_result_355624)
        
        # Call to assert_array_almost_equal(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'tuk1' (line 436)
        tuk1_355626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 34), 'tuk1', False)
        # Getting the type of 'han1' (line 436)
        han1_355627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 40), 'han1', False)
        # Processing the call keyword arguments (line 436)
        kwargs_355628 = {}
        # Getting the type of 'assert_array_almost_equal' (line 436)
        assert_array_almost_equal_355625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 436)
        assert_array_almost_equal_call_result_355629 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), assert_array_almost_equal_355625, *[tuk1_355626, han1_355627], **kwargs_355628)
        
        
        # ################# End of 'test_extremes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_extremes' in the type store
        # Getting the type of 'stypy_return_type' (line 428)
        stypy_return_type_355630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_extremes'
        return stypy_return_type_355630


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 417, 0, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTukey.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTukey' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'TestTukey', TestTukey)
# Declaration of the 'TestGetWindow' class

class TestGetWindow(object, ):

    @norecursion
    def test_boxcar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_boxcar'
        module_type_store = module_type_store.open_function_context('test_boxcar', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_boxcar')
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_boxcar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_boxcar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_boxcar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_boxcar(...)' code ##################

        
        # Assigning a Call to a Name (line 442):
        
        # Call to get_window(...): (line 442)
        # Processing the call arguments (line 442)
        str_355633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 30), 'str', 'boxcar')
        int_355634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 40), 'int')
        # Processing the call keyword arguments (line 442)
        kwargs_355635 = {}
        # Getting the type of 'signal' (line 442)
        signal_355631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 442)
        get_window_355632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 12), signal_355631, 'get_window')
        # Calling get_window(args, kwargs) (line 442)
        get_window_call_result_355636 = invoke(stypy.reporting.localization.Localization(__file__, 442, 12), get_window_355632, *[str_355633, int_355634], **kwargs_355635)
        
        # Assigning a type to the variable 'w' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'w', get_window_call_result_355636)
        
        # Call to assert_array_equal(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'w' (line 443)
        w_355638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 27), 'w', False)
        
        # Call to ones_like(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'w' (line 443)
        w_355641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 43), 'w', False)
        # Processing the call keyword arguments (line 443)
        kwargs_355642 = {}
        # Getting the type of 'np' (line 443)
        np_355639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 30), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 443)
        ones_like_355640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 30), np_355639, 'ones_like')
        # Calling ones_like(args, kwargs) (line 443)
        ones_like_call_result_355643 = invoke(stypy.reporting.localization.Localization(__file__, 443, 30), ones_like_355640, *[w_355641], **kwargs_355642)
        
        # Processing the call keyword arguments (line 443)
        kwargs_355644 = {}
        # Getting the type of 'assert_array_equal' (line 443)
        assert_array_equal_355637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 443)
        assert_array_equal_call_result_355645 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), assert_array_equal_355637, *[w_355638, ones_like_call_result_355643], **kwargs_355644)
        
        
        # Assigning a Call to a Name (line 446):
        
        # Call to get_window(...): (line 446)
        # Processing the call arguments (line 446)
        
        # Obtaining an instance of the builtin type 'tuple' (line 446)
        tuple_355648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 446)
        # Adding element type (line 446)
        str_355649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 31), 'str', 'boxcar')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 31), tuple_355648, str_355649)
        
        int_355650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 43), 'int')
        # Processing the call keyword arguments (line 446)
        kwargs_355651 = {}
        # Getting the type of 'signal' (line 446)
        signal_355646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 446)
        get_window_355647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), signal_355646, 'get_window')
        # Calling get_window(args, kwargs) (line 446)
        get_window_call_result_355652 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), get_window_355647, *[tuple_355648, int_355650], **kwargs_355651)
        
        # Assigning a type to the variable 'w' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'w', get_window_call_result_355652)
        
        # Call to assert_array_equal(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'w' (line 447)
        w_355654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 27), 'w', False)
        
        # Call to ones_like(...): (line 447)
        # Processing the call arguments (line 447)
        # Getting the type of 'w' (line 447)
        w_355657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 43), 'w', False)
        # Processing the call keyword arguments (line 447)
        kwargs_355658 = {}
        # Getting the type of 'np' (line 447)
        np_355655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 30), 'np', False)
        # Obtaining the member 'ones_like' of a type (line 447)
        ones_like_355656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 30), np_355655, 'ones_like')
        # Calling ones_like(args, kwargs) (line 447)
        ones_like_call_result_355659 = invoke(stypy.reporting.localization.Localization(__file__, 447, 30), ones_like_355656, *[w_355657], **kwargs_355658)
        
        # Processing the call keyword arguments (line 447)
        kwargs_355660 = {}
        # Getting the type of 'assert_array_equal' (line 447)
        assert_array_equal_355653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 447)
        assert_array_equal_call_result_355661 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), assert_array_equal_355653, *[w_355654, ones_like_call_result_355659], **kwargs_355660)
        
        
        # ################# End of 'test_boxcar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_boxcar' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_355662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355662)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_boxcar'
        return stypy_return_type_355662


    @norecursion
    def test_cheb_odd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_odd'
        module_type_store = module_type_store.open_function_context('test_cheb_odd', 449, 4, False)
        # Assigning a type to the variable 'self' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_cheb_odd')
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_cheb_odd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_cheb_odd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_odd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_odd(...)' code ##################

        
        # Call to suppress_warnings(...): (line 450)
        # Processing the call keyword arguments (line 450)
        kwargs_355664 = {}
        # Getting the type of 'suppress_warnings' (line 450)
        suppress_warnings_355663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 450)
        suppress_warnings_call_result_355665 = invoke(stypy.reporting.localization.Localization(__file__, 450, 13), suppress_warnings_355663, *[], **kwargs_355664)
        
        with_355666 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 450, 13), suppress_warnings_call_result_355665, 'with parameter', '__enter__', '__exit__')

        if with_355666:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 450)
            enter___355667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 13), suppress_warnings_call_result_355665, '__enter__')
            with_enter_355668 = invoke(stypy.reporting.localization.Localization(__file__, 450, 13), enter___355667)
            # Assigning a type to the variable 'sup' (line 450)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 13), 'sup', with_enter_355668)
            
            # Call to filter(...): (line 451)
            # Processing the call arguments (line 451)
            # Getting the type of 'UserWarning' (line 451)
            UserWarning_355671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'UserWarning', False)
            str_355672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 451)
            kwargs_355673 = {}
            # Getting the type of 'sup' (line 451)
            sup_355669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 451)
            filter_355670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), sup_355669, 'filter')
            # Calling filter(args, kwargs) (line 451)
            filter_call_result_355674 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), filter_355670, *[UserWarning_355671, str_355672], **kwargs_355673)
            
            
            # Assigning a Call to a Name (line 452):
            
            # Call to get_window(...): (line 452)
            # Processing the call arguments (line 452)
            
            # Obtaining an instance of the builtin type 'tuple' (line 452)
            tuple_355677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 452)
            # Adding element type (line 452)
            str_355678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 35), 'str', 'chebwin')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 35), tuple_355677, str_355678)
            # Adding element type (line 452)
            int_355679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 46), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 35), tuple_355677, int_355679)
            
            int_355680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 52), 'int')
            # Processing the call keyword arguments (line 452)
            # Getting the type of 'False' (line 452)
            False_355681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 64), 'False', False)
            keyword_355682 = False_355681
            kwargs_355683 = {'fftbins': keyword_355682}
            # Getting the type of 'signal' (line 452)
            signal_355675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'signal', False)
            # Obtaining the member 'get_window' of a type (line 452)
            get_window_355676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 16), signal_355675, 'get_window')
            # Calling get_window(args, kwargs) (line 452)
            get_window_call_result_355684 = invoke(stypy.reporting.localization.Localization(__file__, 452, 16), get_window_355676, *[tuple_355677, int_355680], **kwargs_355683)
            
            # Assigning a type to the variable 'w' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'w', get_window_call_result_355684)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 450)
            exit___355685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 13), suppress_warnings_call_result_355665, '__exit__')
            with_exit_355686 = invoke(stypy.reporting.localization.Localization(__file__, 450, 13), exit___355685, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 453)
        # Processing the call arguments (line 453)
        # Getting the type of 'w' (line 453)
        w_355688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 34), 'w', False)
        # Getting the type of 'cheb_odd_true' (line 453)
        cheb_odd_true_355689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 37), 'cheb_odd_true', False)
        # Processing the call keyword arguments (line 453)
        int_355690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 60), 'int')
        keyword_355691 = int_355690
        kwargs_355692 = {'decimal': keyword_355691}
        # Getting the type of 'assert_array_almost_equal' (line 453)
        assert_array_almost_equal_355687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 453)
        assert_array_almost_equal_call_result_355693 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), assert_array_almost_equal_355687, *[w_355688, cheb_odd_true_355689], **kwargs_355692)
        
        
        # ################# End of 'test_cheb_odd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_odd' in the type store
        # Getting the type of 'stypy_return_type' (line 449)
        stypy_return_type_355694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355694)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_odd'
        return stypy_return_type_355694


    @norecursion
    def test_cheb_even(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cheb_even'
        module_type_store = module_type_store.open_function_context('test_cheb_even', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_cheb_even')
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_cheb_even.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_cheb_even', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cheb_even', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cheb_even(...)' code ##################

        
        # Call to suppress_warnings(...): (line 456)
        # Processing the call keyword arguments (line 456)
        kwargs_355696 = {}
        # Getting the type of 'suppress_warnings' (line 456)
        suppress_warnings_355695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 456)
        suppress_warnings_call_result_355697 = invoke(stypy.reporting.localization.Localization(__file__, 456, 13), suppress_warnings_355695, *[], **kwargs_355696)
        
        with_355698 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 456, 13), suppress_warnings_call_result_355697, 'with parameter', '__enter__', '__exit__')

        if with_355698:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 456)
            enter___355699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 13), suppress_warnings_call_result_355697, '__enter__')
            with_enter_355700 = invoke(stypy.reporting.localization.Localization(__file__, 456, 13), enter___355699)
            # Assigning a type to the variable 'sup' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'sup', with_enter_355700)
            
            # Call to filter(...): (line 457)
            # Processing the call arguments (line 457)
            # Getting the type of 'UserWarning' (line 457)
            UserWarning_355703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'UserWarning', False)
            str_355704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 36), 'str', 'This window is not suitable')
            # Processing the call keyword arguments (line 457)
            kwargs_355705 = {}
            # Getting the type of 'sup' (line 457)
            sup_355701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 457)
            filter_355702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), sup_355701, 'filter')
            # Calling filter(args, kwargs) (line 457)
            filter_call_result_355706 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), filter_355702, *[UserWarning_355703, str_355704], **kwargs_355705)
            
            
            # Assigning a Call to a Name (line 458):
            
            # Call to get_window(...): (line 458)
            # Processing the call arguments (line 458)
            
            # Obtaining an instance of the builtin type 'tuple' (line 458)
            tuple_355709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 458)
            # Adding element type (line 458)
            str_355710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 35), 'str', 'chebwin')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 35), tuple_355709, str_355710)
            # Adding element type (line 458)
            int_355711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 46), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 35), tuple_355709, int_355711)
            
            int_355712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 51), 'int')
            # Processing the call keyword arguments (line 458)
            # Getting the type of 'False' (line 458)
            False_355713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 63), 'False', False)
            keyword_355714 = False_355713
            kwargs_355715 = {'fftbins': keyword_355714}
            # Getting the type of 'signal' (line 458)
            signal_355707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'signal', False)
            # Obtaining the member 'get_window' of a type (line 458)
            get_window_355708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 16), signal_355707, 'get_window')
            # Calling get_window(args, kwargs) (line 458)
            get_window_call_result_355716 = invoke(stypy.reporting.localization.Localization(__file__, 458, 16), get_window_355708, *[tuple_355709, int_355712], **kwargs_355715)
            
            # Assigning a type to the variable 'w' (line 458)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'w', get_window_call_result_355716)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 456)
            exit___355717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 13), suppress_warnings_call_result_355697, '__exit__')
            with_exit_355718 = invoke(stypy.reporting.localization.Localization(__file__, 456, 13), exit___355717, None, None, None)

        
        # Call to assert_array_almost_equal(...): (line 459)
        # Processing the call arguments (line 459)
        # Getting the type of 'w' (line 459)
        w_355720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 34), 'w', False)
        # Getting the type of 'cheb_even_true' (line 459)
        cheb_even_true_355721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 37), 'cheb_even_true', False)
        # Processing the call keyword arguments (line 459)
        int_355722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 61), 'int')
        keyword_355723 = int_355722
        kwargs_355724 = {'decimal': keyword_355723}
        # Getting the type of 'assert_array_almost_equal' (line 459)
        assert_array_almost_equal_355719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 459)
        assert_array_almost_equal_call_result_355725 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), assert_array_almost_equal_355719, *[w_355720, cheb_even_true_355721], **kwargs_355724)
        
        
        # ################# End of 'test_cheb_even(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cheb_even' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_355726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cheb_even'
        return stypy_return_type_355726


    @norecursion
    def test_kaiser_float(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kaiser_float'
        module_type_store = module_type_store.open_function_context('test_kaiser_float', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_kaiser_float')
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_kaiser_float.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_kaiser_float', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kaiser_float', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kaiser_float(...)' code ##################

        
        # Assigning a Call to a Name (line 462):
        
        # Call to get_window(...): (line 462)
        # Processing the call arguments (line 462)
        float_355729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 33), 'float')
        int_355730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 38), 'int')
        # Processing the call keyword arguments (line 462)
        kwargs_355731 = {}
        # Getting the type of 'signal' (line 462)
        signal_355727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 462)
        get_window_355728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 15), signal_355727, 'get_window')
        # Calling get_window(args, kwargs) (line 462)
        get_window_call_result_355732 = invoke(stypy.reporting.localization.Localization(__file__, 462, 15), get_window_355728, *[float_355729, int_355730], **kwargs_355731)
        
        # Assigning a type to the variable 'win1' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'win1', get_window_call_result_355732)
        
        # Assigning a Call to a Name (line 463):
        
        # Call to kaiser(...): (line 463)
        # Processing the call arguments (line 463)
        int_355735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 29), 'int')
        float_355736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 33), 'float')
        # Getting the type of 'False' (line 463)
        False_355737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 38), 'False', False)
        # Processing the call keyword arguments (line 463)
        kwargs_355738 = {}
        # Getting the type of 'signal' (line 463)
        signal_355733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'signal', False)
        # Obtaining the member 'kaiser' of a type (line 463)
        kaiser_355734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 15), signal_355733, 'kaiser')
        # Calling kaiser(args, kwargs) (line 463)
        kaiser_call_result_355739 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), kaiser_355734, *[int_355735, float_355736, False_355737], **kwargs_355738)
        
        # Assigning a type to the variable 'win2' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'win2', kaiser_call_result_355739)
        
        # Call to assert_allclose(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'win1' (line 464)
        win1_355741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'win1', False)
        # Getting the type of 'win2' (line 464)
        win2_355742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 30), 'win2', False)
        # Processing the call keyword arguments (line 464)
        kwargs_355743 = {}
        # Getting the type of 'assert_allclose' (line 464)
        assert_allclose_355740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 464)
        assert_allclose_call_result_355744 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), assert_allclose_355740, *[win1_355741, win2_355742], **kwargs_355743)
        
        
        # ################# End of 'test_kaiser_float(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kaiser_float' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_355745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kaiser_float'
        return stypy_return_type_355745


    @norecursion
    def test_invalid_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_invalid_inputs'
        module_type_store = module_type_store.open_function_context('test_invalid_inputs', 466, 4, False)
        # Assigning a type to the variable 'self' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_invalid_inputs')
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_invalid_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_invalid_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_invalid_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_invalid_inputs(...)' code ##################

        
        # Call to assert_raises(...): (line 468)
        # Processing the call arguments (line 468)
        # Getting the type of 'ValueError' (line 468)
        ValueError_355747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 22), 'ValueError', False)
        # Getting the type of 'signal' (line 468)
        signal_355748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 34), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 468)
        get_window_355749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 34), signal_355748, 'get_window')
        
        # Call to set(...): (line 468)
        # Processing the call arguments (line 468)
        str_355751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 57), 'str', 'hann')
        # Processing the call keyword arguments (line 468)
        kwargs_355752 = {}
        # Getting the type of 'set' (line 468)
        set_355750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 53), 'set', False)
        # Calling set(args, kwargs) (line 468)
        set_call_result_355753 = invoke(stypy.reporting.localization.Localization(__file__, 468, 53), set_355750, *[str_355751], **kwargs_355752)
        
        int_355754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 66), 'int')
        # Processing the call keyword arguments (line 468)
        kwargs_355755 = {}
        # Getting the type of 'assert_raises' (line 468)
        assert_raises_355746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 468)
        assert_raises_call_result_355756 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), assert_raises_355746, *[ValueError_355747, get_window_355749, set_call_result_355753, int_355754], **kwargs_355755)
        
        
        # Call to assert_raises(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'ValueError' (line 471)
        ValueError_355758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'ValueError', False)
        # Getting the type of 'signal' (line 471)
        signal_355759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 471)
        get_window_355760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 34), signal_355759, 'get_window')
        str_355761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 53), 'str', 'broken')
        int_355762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 63), 'int')
        # Processing the call keyword arguments (line 471)
        kwargs_355763 = {}
        # Getting the type of 'assert_raises' (line 471)
        assert_raises_355757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 471)
        assert_raises_call_result_355764 = invoke(stypy.reporting.localization.Localization(__file__, 471, 8), assert_raises_355757, *[ValueError_355758, get_window_355760, str_355761, int_355762], **kwargs_355763)
        
        
        # ################# End of 'test_invalid_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_invalid_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 466)
        stypy_return_type_355765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_invalid_inputs'
        return stypy_return_type_355765


    @norecursion
    def test_array_as_window(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_array_as_window'
        module_type_store = module_type_store.open_function_context('test_array_as_window', 473, 4, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_localization', localization)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_function_name', 'TestGetWindow.test_array_as_window')
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_param_names_list', [])
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGetWindow.test_array_as_window.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.test_array_as_window', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_array_as_window', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_array_as_window(...)' code ##################

        
        # Assigning a Num to a Name (line 475):
        int_355766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 19), 'int')
        # Assigning a type to the variable 'osfactor' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'osfactor', int_355766)
        
        # Assigning a Call to a Name (line 476):
        
        # Call to arange(...): (line 476)
        # Processing the call arguments (line 476)
        int_355769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 24), 'int')
        # Processing the call keyword arguments (line 476)
        kwargs_355770 = {}
        # Getting the type of 'np' (line 476)
        np_355767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 14), 'np', False)
        # Obtaining the member 'arange' of a type (line 476)
        arange_355768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 14), np_355767, 'arange')
        # Calling arange(args, kwargs) (line 476)
        arange_call_result_355771 = invoke(stypy.reporting.localization.Localization(__file__, 476, 14), arange_355768, *[int_355769], **kwargs_355770)
        
        # Assigning a type to the variable 'sig' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'sig', arange_call_result_355771)
        
        # Assigning a Call to a Name (line 478):
        
        # Call to get_window(...): (line 478)
        # Processing the call arguments (line 478)
        
        # Obtaining an instance of the builtin type 'tuple' (line 478)
        tuple_355774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 478)
        # Adding element type (line 478)
        str_355775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 33), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), tuple_355774, str_355775)
        # Adding element type (line 478)
        float_355776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 33), tuple_355774, float_355776)
        
        # Getting the type of 'osfactor' (line 478)
        osfactor_355777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 49), 'osfactor', False)
        int_355778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 61), 'int')
        # Applying the binary operator '//' (line 478)
        result_floordiv_355779 = python_operator(stypy.reporting.localization.Localization(__file__, 478, 49), '//', osfactor_355777, int_355778)
        
        # Processing the call keyword arguments (line 478)
        kwargs_355780 = {}
        # Getting the type of 'signal' (line 478)
        signal_355772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 14), 'signal', False)
        # Obtaining the member 'get_window' of a type (line 478)
        get_window_355773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 14), signal_355772, 'get_window')
        # Calling get_window(args, kwargs) (line 478)
        get_window_call_result_355781 = invoke(stypy.reporting.localization.Localization(__file__, 478, 14), get_window_355773, *[tuple_355774, result_floordiv_355779], **kwargs_355780)
        
        # Assigning a type to the variable 'win' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'win', get_window_call_result_355781)
        
        # Call to assert_raises(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'ValueError' (line 479)
        ValueError_355783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 22), 'ValueError', False)
        # Getting the type of 'signal' (line 479)
        signal_355784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 34), 'signal', False)
        # Obtaining the member 'resample' of a type (line 479)
        resample_355785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 34), signal_355784, 'resample')
        
        # Obtaining an instance of the builtin type 'tuple' (line 480)
        tuple_355786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 480)
        # Adding element type (line 480)
        # Getting the type of 'sig' (line 480)
        sig_355787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 23), 'sig', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 23), tuple_355786, sig_355787)
        # Adding element type (line 480)
        
        # Call to len(...): (line 480)
        # Processing the call arguments (line 480)
        # Getting the type of 'sig' (line 480)
        sig_355789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 32), 'sig', False)
        # Processing the call keyword arguments (line 480)
        kwargs_355790 = {}
        # Getting the type of 'len' (line 480)
        len_355788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 28), 'len', False)
        # Calling len(args, kwargs) (line 480)
        len_call_result_355791 = invoke(stypy.reporting.localization.Localization(__file__, 480, 28), len_355788, *[sig_355789], **kwargs_355790)
        
        # Getting the type of 'osfactor' (line 480)
        osfactor_355792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 39), 'osfactor', False)
        # Applying the binary operator '*' (line 480)
        result_mul_355793 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 28), '*', len_call_result_355791, osfactor_355792)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 23), tuple_355786, result_mul_355793)
        
        
        # Obtaining an instance of the builtin type 'dict' (line 480)
        dict_355794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 50), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 480)
        # Adding element type (key, value) (line 480)
        str_355795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 51), 'str', 'window')
        # Getting the type of 'win' (line 480)
        win_355796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 61), 'win', False)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 480, 50), dict_355794, (str_355795, win_355796))
        
        # Processing the call keyword arguments (line 479)
        kwargs_355797 = {}
        # Getting the type of 'assert_raises' (line 479)
        assert_raises_355782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 479)
        assert_raises_call_result_355798 = invoke(stypy.reporting.localization.Localization(__file__, 479, 8), assert_raises_355782, *[ValueError_355783, resample_355785, tuple_355786, dict_355794], **kwargs_355797)
        
        
        # ################# End of 'test_array_as_window(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_array_as_window' in the type store
        # Getting the type of 'stypy_return_type' (line 473)
        stypy_return_type_355799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_355799)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_array_as_window'
        return stypy_return_type_355799


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 439, 0, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGetWindow.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGetWindow' (line 439)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 0), 'TestGetWindow', TestGetWindow)

@norecursion
def test_windowfunc_basics(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_windowfunc_basics'
    module_type_store = module_type_store.open_function_context('test_windowfunc_basics', 483, 0, False)
    
    # Passed parameters checking function
    test_windowfunc_basics.stypy_localization = localization
    test_windowfunc_basics.stypy_type_of_self = None
    test_windowfunc_basics.stypy_type_store = module_type_store
    test_windowfunc_basics.stypy_function_name = 'test_windowfunc_basics'
    test_windowfunc_basics.stypy_param_names_list = []
    test_windowfunc_basics.stypy_varargs_param_name = None
    test_windowfunc_basics.stypy_kwargs_param_name = None
    test_windowfunc_basics.stypy_call_defaults = defaults
    test_windowfunc_basics.stypy_call_varargs = varargs
    test_windowfunc_basics.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_windowfunc_basics', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_windowfunc_basics', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_windowfunc_basics(...)' code ##################

    
    # Getting the type of 'window_funcs' (line 484)
    window_funcs_355800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 31), 'window_funcs')
    # Testing the type of a for loop iterable (line 484)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 484, 4), window_funcs_355800)
    # Getting the type of the for loop variable (line 484)
    for_loop_var_355801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 484, 4), window_funcs_355800)
    # Assigning a type to the variable 'window_name' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'window_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 4), for_loop_var_355801))
    # Assigning a type to the variable 'params' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'params', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 4), for_loop_var_355801))
    # SSA begins for a for statement (line 484)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 485):
    
    # Call to getattr(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'signal' (line 485)
    signal_355803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 25), 'signal', False)
    # Getting the type of 'window_name' (line 485)
    window_name_355804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 33), 'window_name', False)
    # Processing the call keyword arguments (line 485)
    kwargs_355805 = {}
    # Getting the type of 'getattr' (line 485)
    getattr_355802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 17), 'getattr', False)
    # Calling getattr(args, kwargs) (line 485)
    getattr_call_result_355806 = invoke(stypy.reporting.localization.Localization(__file__, 485, 17), getattr_355802, *[signal_355803, window_name_355804], **kwargs_355805)
    
    # Assigning a type to the variable 'window' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'window', getattr_call_result_355806)
    
    # Call to suppress_warnings(...): (line 486)
    # Processing the call keyword arguments (line 486)
    kwargs_355808 = {}
    # Getting the type of 'suppress_warnings' (line 486)
    suppress_warnings_355807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 13), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 486)
    suppress_warnings_call_result_355809 = invoke(stypy.reporting.localization.Localization(__file__, 486, 13), suppress_warnings_355807, *[], **kwargs_355808)
    
    with_355810 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 486, 13), suppress_warnings_call_result_355809, 'with parameter', '__enter__', '__exit__')

    if with_355810:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 486)
        enter___355811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 13), suppress_warnings_call_result_355809, '__enter__')
        with_enter_355812 = invoke(stypy.reporting.localization.Localization(__file__, 486, 13), enter___355811)
        # Assigning a type to the variable 'sup' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 13), 'sup', with_enter_355812)
        
        # Call to filter(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'UserWarning' (line 487)
        UserWarning_355815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 23), 'UserWarning', False)
        str_355816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 36), 'str', 'This window is not suitable')
        # Processing the call keyword arguments (line 487)
        kwargs_355817 = {}
        # Getting the type of 'sup' (line 487)
        sup_355813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'sup', False)
        # Obtaining the member 'filter' of a type (line 487)
        filter_355814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), sup_355813, 'filter')
        # Calling filter(args, kwargs) (line 487)
        filter_call_result_355818 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), filter_355814, *[UserWarning_355815, str_355816], **kwargs_355817)
        
        
        # Assigning a Call to a Name (line 489):
        
        # Call to window(...): (line 489)
        # Processing the call arguments (line 489)
        int_355820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 24), 'int')
        # Getting the type of 'params' (line 489)
        params_355821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 28), 'params', False)
        # Processing the call keyword arguments (line 489)
        # Getting the type of 'True' (line 489)
        True_355822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 40), 'True', False)
        keyword_355823 = True_355822
        kwargs_355824 = {'sym': keyword_355823}
        # Getting the type of 'window' (line 489)
        window_355819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 17), 'window', False)
        # Calling window(args, kwargs) (line 489)
        window_call_result_355825 = invoke(stypy.reporting.localization.Localization(__file__, 489, 17), window_355819, *[int_355820, params_355821], **kwargs_355824)
        
        # Assigning a type to the variable 'w1' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'w1', window_call_result_355825)
        
        # Assigning a Call to a Name (line 490):
        
        # Call to window(...): (line 490)
        # Processing the call arguments (line 490)
        int_355827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 24), 'int')
        # Getting the type of 'params' (line 490)
        params_355828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 28), 'params', False)
        # Processing the call keyword arguments (line 490)
        # Getting the type of 'False' (line 490)
        False_355829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 40), 'False', False)
        keyword_355830 = False_355829
        kwargs_355831 = {'sym': keyword_355830}
        # Getting the type of 'window' (line 490)
        window_355826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 17), 'window', False)
        # Calling window(args, kwargs) (line 490)
        window_call_result_355832 = invoke(stypy.reporting.localization.Localization(__file__, 490, 17), window_355826, *[int_355827, params_355828], **kwargs_355831)
        
        # Assigning a type to the variable 'w2' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'w2', window_call_result_355832)
        
        # Call to assert_array_almost_equal(...): (line 491)
        # Processing the call arguments (line 491)
        
        # Obtaining the type of the subscript
        int_355834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 42), 'int')
        slice_355835 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 491, 38), None, int_355834, None)
        # Getting the type of 'w1' (line 491)
        w1_355836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 38), 'w1', False)
        # Obtaining the member '__getitem__' of a type (line 491)
        getitem___355837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 38), w1_355836, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 491)
        subscript_call_result_355838 = invoke(stypy.reporting.localization.Localization(__file__, 491, 38), getitem___355837, slice_355835)
        
        # Getting the type of 'w2' (line 491)
        w2_355839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 47), 'w2', False)
        # Processing the call keyword arguments (line 491)
        kwargs_355840 = {}
        # Getting the type of 'assert_array_almost_equal' (line 491)
        assert_array_almost_equal_355833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 491)
        assert_array_almost_equal_call_result_355841 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), assert_array_almost_equal_355833, *[subscript_call_result_355838, w2_355839], **kwargs_355840)
        
        
        # Assigning a Call to a Name (line 493):
        
        # Call to window(...): (line 493)
        # Processing the call arguments (line 493)
        int_355843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 24), 'int')
        # Getting the type of 'params' (line 493)
        params_355844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 28), 'params', False)
        # Processing the call keyword arguments (line 493)
        # Getting the type of 'True' (line 493)
        True_355845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 40), 'True', False)
        keyword_355846 = True_355845
        kwargs_355847 = {'sym': keyword_355846}
        # Getting the type of 'window' (line 493)
        window_355842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 17), 'window', False)
        # Calling window(args, kwargs) (line 493)
        window_call_result_355848 = invoke(stypy.reporting.localization.Localization(__file__, 493, 17), window_355842, *[int_355843, params_355844], **kwargs_355847)
        
        # Assigning a type to the variable 'w1' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'w1', window_call_result_355848)
        
        # Assigning a Call to a Name (line 494):
        
        # Call to window(...): (line 494)
        # Processing the call arguments (line 494)
        int_355850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 24), 'int')
        # Getting the type of 'params' (line 494)
        params_355851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 28), 'params', False)
        # Processing the call keyword arguments (line 494)
        # Getting the type of 'False' (line 494)
        False_355852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 40), 'False', False)
        keyword_355853 = False_355852
        kwargs_355854 = {'sym': keyword_355853}
        # Getting the type of 'window' (line 494)
        window_355849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 17), 'window', False)
        # Calling window(args, kwargs) (line 494)
        window_call_result_355855 = invoke(stypy.reporting.localization.Localization(__file__, 494, 17), window_355849, *[int_355850, params_355851], **kwargs_355854)
        
        # Assigning a type to the variable 'w2' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'w2', window_call_result_355855)
        
        # Call to assert_array_almost_equal(...): (line 495)
        # Processing the call arguments (line 495)
        
        # Obtaining the type of the subscript
        int_355857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 42), 'int')
        slice_355858 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 495, 38), None, int_355857, None)
        # Getting the type of 'w1' (line 495)
        w1_355859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 38), 'w1', False)
        # Obtaining the member '__getitem__' of a type (line 495)
        getitem___355860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 38), w1_355859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 495)
        subscript_call_result_355861 = invoke(stypy.reporting.localization.Localization(__file__, 495, 38), getitem___355860, slice_355858)
        
        # Getting the type of 'w2' (line 495)
        w2_355862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 47), 'w2', False)
        # Processing the call keyword arguments (line 495)
        kwargs_355863 = {}
        # Getting the type of 'assert_array_almost_equal' (line 495)
        assert_array_almost_equal_355856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 495)
        assert_array_almost_equal_call_result_355864 = invoke(stypy.reporting.localization.Localization(__file__, 495, 12), assert_array_almost_equal_355856, *[subscript_call_result_355861, w2_355862], **kwargs_355863)
        
        
        # Call to assert_equal(...): (line 498)
        # Processing the call arguments (line 498)
        
        # Call to len(...): (line 498)
        # Processing the call arguments (line 498)
        
        # Call to window(...): (line 498)
        # Processing the call arguments (line 498)
        int_355868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 36), 'int')
        # Getting the type of 'params' (line 498)
        params_355869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 40), 'params', False)
        # Processing the call keyword arguments (line 498)
        # Getting the type of 'True' (line 498)
        True_355870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 52), 'True', False)
        keyword_355871 = True_355870
        kwargs_355872 = {'sym': keyword_355871}
        # Getting the type of 'window' (line 498)
        window_355867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 29), 'window', False)
        # Calling window(args, kwargs) (line 498)
        window_call_result_355873 = invoke(stypy.reporting.localization.Localization(__file__, 498, 29), window_355867, *[int_355868, params_355869], **kwargs_355872)
        
        # Processing the call keyword arguments (line 498)
        kwargs_355874 = {}
        # Getting the type of 'len' (line 498)
        len_355866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'len', False)
        # Calling len(args, kwargs) (line 498)
        len_call_result_355875 = invoke(stypy.reporting.localization.Localization(__file__, 498, 25), len_355866, *[window_call_result_355873], **kwargs_355874)
        
        int_355876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 60), 'int')
        # Processing the call keyword arguments (line 498)
        kwargs_355877 = {}
        # Getting the type of 'assert_equal' (line 498)
        assert_equal_355865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 498)
        assert_equal_call_result_355878 = invoke(stypy.reporting.localization.Localization(__file__, 498, 12), assert_equal_355865, *[len_call_result_355875, int_355876], **kwargs_355877)
        
        
        # Call to assert_equal(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to len(...): (line 499)
        # Processing the call arguments (line 499)
        
        # Call to window(...): (line 499)
        # Processing the call arguments (line 499)
        int_355882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 36), 'int')
        # Getting the type of 'params' (line 499)
        params_355883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 40), 'params', False)
        # Processing the call keyword arguments (line 499)
        # Getting the type of 'False' (line 499)
        False_355884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 52), 'False', False)
        keyword_355885 = False_355884
        kwargs_355886 = {'sym': keyword_355885}
        # Getting the type of 'window' (line 499)
        window_355881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 29), 'window', False)
        # Calling window(args, kwargs) (line 499)
        window_call_result_355887 = invoke(stypy.reporting.localization.Localization(__file__, 499, 29), window_355881, *[int_355882, params_355883], **kwargs_355886)
        
        # Processing the call keyword arguments (line 499)
        kwargs_355888 = {}
        # Getting the type of 'len' (line 499)
        len_355880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 25), 'len', False)
        # Calling len(args, kwargs) (line 499)
        len_call_result_355889 = invoke(stypy.reporting.localization.Localization(__file__, 499, 25), len_355880, *[window_call_result_355887], **kwargs_355888)
        
        int_355890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 61), 'int')
        # Processing the call keyword arguments (line 499)
        kwargs_355891 = {}
        # Getting the type of 'assert_equal' (line 499)
        assert_equal_355879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 499)
        assert_equal_call_result_355892 = invoke(stypy.reporting.localization.Localization(__file__, 499, 12), assert_equal_355879, *[len_call_result_355889, int_355890], **kwargs_355891)
        
        
        # Call to assert_equal(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to len(...): (line 500)
        # Processing the call arguments (line 500)
        
        # Call to window(...): (line 500)
        # Processing the call arguments (line 500)
        int_355896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 36), 'int')
        # Getting the type of 'params' (line 500)
        params_355897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 40), 'params', False)
        # Processing the call keyword arguments (line 500)
        # Getting the type of 'True' (line 500)
        True_355898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 52), 'True', False)
        keyword_355899 = True_355898
        kwargs_355900 = {'sym': keyword_355899}
        # Getting the type of 'window' (line 500)
        window_355895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'window', False)
        # Calling window(args, kwargs) (line 500)
        window_call_result_355901 = invoke(stypy.reporting.localization.Localization(__file__, 500, 29), window_355895, *[int_355896, params_355897], **kwargs_355900)
        
        # Processing the call keyword arguments (line 500)
        kwargs_355902 = {}
        # Getting the type of 'len' (line 500)
        len_355894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 25), 'len', False)
        # Calling len(args, kwargs) (line 500)
        len_call_result_355903 = invoke(stypy.reporting.localization.Localization(__file__, 500, 25), len_355894, *[window_call_result_355901], **kwargs_355902)
        
        int_355904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 60), 'int')
        # Processing the call keyword arguments (line 500)
        kwargs_355905 = {}
        # Getting the type of 'assert_equal' (line 500)
        assert_equal_355893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 500)
        assert_equal_call_result_355906 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), assert_equal_355893, *[len_call_result_355903, int_355904], **kwargs_355905)
        
        
        # Call to assert_equal(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Call to len(...): (line 501)
        # Processing the call arguments (line 501)
        
        # Call to window(...): (line 501)
        # Processing the call arguments (line 501)
        int_355910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 36), 'int')
        # Getting the type of 'params' (line 501)
        params_355911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 40), 'params', False)
        # Processing the call keyword arguments (line 501)
        # Getting the type of 'False' (line 501)
        False_355912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 52), 'False', False)
        keyword_355913 = False_355912
        kwargs_355914 = {'sym': keyword_355913}
        # Getting the type of 'window' (line 501)
        window_355909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 29), 'window', False)
        # Calling window(args, kwargs) (line 501)
        window_call_result_355915 = invoke(stypy.reporting.localization.Localization(__file__, 501, 29), window_355909, *[int_355910, params_355911], **kwargs_355914)
        
        # Processing the call keyword arguments (line 501)
        kwargs_355916 = {}
        # Getting the type of 'len' (line 501)
        len_355908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'len', False)
        # Calling len(args, kwargs) (line 501)
        len_call_result_355917 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), len_355908, *[window_call_result_355915], **kwargs_355916)
        
        int_355918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 61), 'int')
        # Processing the call keyword arguments (line 501)
        kwargs_355919 = {}
        # Getting the type of 'assert_equal' (line 501)
        assert_equal_355907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 501)
        assert_equal_call_result_355920 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), assert_equal_355907, *[len_call_result_355917, int_355918], **kwargs_355919)
        
        
        # Call to assert_raises(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'ValueError' (line 504)
        ValueError_355922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 26), 'ValueError', False)
        # Getting the type of 'window' (line 504)
        window_355923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 38), 'window', False)
        float_355924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 46), 'float')
        # Getting the type of 'params' (line 504)
        params_355925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 52), 'params', False)
        # Processing the call keyword arguments (line 504)
        kwargs_355926 = {}
        # Getting the type of 'assert_raises' (line 504)
        assert_raises_355921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 504)
        assert_raises_call_result_355927 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), assert_raises_355921, *[ValueError_355922, window_355923, float_355924, params_355925], **kwargs_355926)
        
        
        # Call to assert_raises(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'ValueError' (line 505)
        ValueError_355929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 26), 'ValueError', False)
        # Getting the type of 'window' (line 505)
        window_355930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 38), 'window', False)
        int_355931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 46), 'int')
        # Getting the type of 'params' (line 505)
        params_355932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 51), 'params', False)
        # Processing the call keyword arguments (line 505)
        kwargs_355933 = {}
        # Getting the type of 'assert_raises' (line 505)
        assert_raises_355928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 505)
        assert_raises_call_result_355934 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), assert_raises_355928, *[ValueError_355929, window_355930, int_355931, params_355932], **kwargs_355933)
        
        
        # Call to assert_array_equal(...): (line 508)
        # Processing the call arguments (line 508)
        
        # Call to window(...): (line 508)
        # Processing the call arguments (line 508)
        int_355937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 38), 'int')
        # Getting the type of 'params' (line 508)
        params_355938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 42), 'params', False)
        # Processing the call keyword arguments (line 508)
        # Getting the type of 'True' (line 508)
        True_355939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 54), 'True', False)
        keyword_355940 = True_355939
        kwargs_355941 = {'sym': keyword_355940}
        # Getting the type of 'window' (line 508)
        window_355936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 31), 'window', False)
        # Calling window(args, kwargs) (line 508)
        window_call_result_355942 = invoke(stypy.reporting.localization.Localization(__file__, 508, 31), window_355936, *[int_355937, params_355938], **kwargs_355941)
        
        
        # Obtaining an instance of the builtin type 'list' (line 508)
        list_355943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 508)
        
        # Processing the call keyword arguments (line 508)
        kwargs_355944 = {}
        # Getting the type of 'assert_array_equal' (line 508)
        assert_array_equal_355935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 508)
        assert_array_equal_call_result_355945 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), assert_array_equal_355935, *[window_call_result_355942, list_355943], **kwargs_355944)
        
        
        # Call to assert_array_equal(...): (line 509)
        # Processing the call arguments (line 509)
        
        # Call to window(...): (line 509)
        # Processing the call arguments (line 509)
        int_355948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 38), 'int')
        # Getting the type of 'params' (line 509)
        params_355949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 42), 'params', False)
        # Processing the call keyword arguments (line 509)
        # Getting the type of 'False' (line 509)
        False_355950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 54), 'False', False)
        keyword_355951 = False_355950
        kwargs_355952 = {'sym': keyword_355951}
        # Getting the type of 'window' (line 509)
        window_355947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 31), 'window', False)
        # Calling window(args, kwargs) (line 509)
        window_call_result_355953 = invoke(stypy.reporting.localization.Localization(__file__, 509, 31), window_355947, *[int_355948, params_355949], **kwargs_355952)
        
        
        # Obtaining an instance of the builtin type 'list' (line 509)
        list_355954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 509)
        
        # Processing the call keyword arguments (line 509)
        kwargs_355955 = {}
        # Getting the type of 'assert_array_equal' (line 509)
        assert_array_equal_355946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 509)
        assert_array_equal_call_result_355956 = invoke(stypy.reporting.localization.Localization(__file__, 509, 12), assert_array_equal_355946, *[window_call_result_355953, list_355954], **kwargs_355955)
        
        
        # Call to assert_array_equal(...): (line 510)
        # Processing the call arguments (line 510)
        
        # Call to window(...): (line 510)
        # Processing the call arguments (line 510)
        int_355959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 38), 'int')
        # Getting the type of 'params' (line 510)
        params_355960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 42), 'params', False)
        # Processing the call keyword arguments (line 510)
        # Getting the type of 'True' (line 510)
        True_355961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 54), 'True', False)
        keyword_355962 = True_355961
        kwargs_355963 = {'sym': keyword_355962}
        # Getting the type of 'window' (line 510)
        window_355958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 31), 'window', False)
        # Calling window(args, kwargs) (line 510)
        window_call_result_355964 = invoke(stypy.reporting.localization.Localization(__file__, 510, 31), window_355958, *[int_355959, params_355960], **kwargs_355963)
        
        
        # Obtaining an instance of the builtin type 'list' (line 510)
        list_355965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 510)
        # Adding element type (line 510)
        int_355966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 510, 61), list_355965, int_355966)
        
        # Processing the call keyword arguments (line 510)
        kwargs_355967 = {}
        # Getting the type of 'assert_array_equal' (line 510)
        assert_array_equal_355957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 510)
        assert_array_equal_call_result_355968 = invoke(stypy.reporting.localization.Localization(__file__, 510, 12), assert_array_equal_355957, *[window_call_result_355964, list_355965], **kwargs_355967)
        
        
        # Call to assert_array_equal(...): (line 511)
        # Processing the call arguments (line 511)
        
        # Call to window(...): (line 511)
        # Processing the call arguments (line 511)
        int_355971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 38), 'int')
        # Getting the type of 'params' (line 511)
        params_355972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 42), 'params', False)
        # Processing the call keyword arguments (line 511)
        # Getting the type of 'False' (line 511)
        False_355973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 54), 'False', False)
        keyword_355974 = False_355973
        kwargs_355975 = {'sym': keyword_355974}
        # Getting the type of 'window' (line 511)
        window_355970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 31), 'window', False)
        # Calling window(args, kwargs) (line 511)
        window_call_result_355976 = invoke(stypy.reporting.localization.Localization(__file__, 511, 31), window_355970, *[int_355971, params_355972], **kwargs_355975)
        
        
        # Obtaining an instance of the builtin type 'list' (line 511)
        list_355977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 511)
        # Adding element type (line 511)
        int_355978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 62), list_355977, int_355978)
        
        # Processing the call keyword arguments (line 511)
        kwargs_355979 = {}
        # Getting the type of 'assert_array_equal' (line 511)
        assert_array_equal_355969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 511)
        assert_array_equal_call_result_355980 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), assert_array_equal_355969, *[window_call_result_355976, list_355977], **kwargs_355979)
        
        
        # Call to assert_(...): (line 514)
        # Processing the call arguments (line 514)
        
        
        # Call to window(...): (line 514)
        # Processing the call arguments (line 514)
        int_355983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 27), 'int')
        # Getting the type of 'params' (line 514)
        params_355984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 31), 'params', False)
        # Processing the call keyword arguments (line 514)
        # Getting the type of 'True' (line 514)
        True_355985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 43), 'True', False)
        keyword_355986 = True_355985
        kwargs_355987 = {'sym': keyword_355986}
        # Getting the type of 'window' (line 514)
        window_355982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'window', False)
        # Calling window(args, kwargs) (line 514)
        window_call_result_355988 = invoke(stypy.reporting.localization.Localization(__file__, 514, 20), window_355982, *[int_355983, params_355984], **kwargs_355987)
        
        # Obtaining the member 'dtype' of a type (line 514)
        dtype_355989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 20), window_call_result_355988, 'dtype')
        str_355990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 58), 'str', 'float')
        # Applying the binary operator '==' (line 514)
        result_eq_355991 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 20), '==', dtype_355989, str_355990)
        
        # Processing the call keyword arguments (line 514)
        kwargs_355992 = {}
        # Getting the type of 'assert_' (line 514)
        assert__355981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 514)
        assert__call_result_355993 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), assert__355981, *[result_eq_355991], **kwargs_355992)
        
        
        # Call to assert_(...): (line 515)
        # Processing the call arguments (line 515)
        
        
        # Call to window(...): (line 515)
        # Processing the call arguments (line 515)
        int_355996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 27), 'int')
        # Getting the type of 'params' (line 515)
        params_355997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 31), 'params', False)
        # Processing the call keyword arguments (line 515)
        # Getting the type of 'False' (line 515)
        False_355998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 43), 'False', False)
        keyword_355999 = False_355998
        kwargs_356000 = {'sym': keyword_355999}
        # Getting the type of 'window' (line 515)
        window_355995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'window', False)
        # Calling window(args, kwargs) (line 515)
        window_call_result_356001 = invoke(stypy.reporting.localization.Localization(__file__, 515, 20), window_355995, *[int_355996, params_355997], **kwargs_356000)
        
        # Obtaining the member 'dtype' of a type (line 515)
        dtype_356002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 20), window_call_result_356001, 'dtype')
        str_356003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 59), 'str', 'float')
        # Applying the binary operator '==' (line 515)
        result_eq_356004 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 20), '==', dtype_356002, str_356003)
        
        # Processing the call keyword arguments (line 515)
        kwargs_356005 = {}
        # Getting the type of 'assert_' (line 515)
        assert__355994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 515)
        assert__call_result_356006 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), assert__355994, *[result_eq_356004], **kwargs_356005)
        
        
        # Call to assert_(...): (line 516)
        # Processing the call arguments (line 516)
        
        
        # Call to window(...): (line 516)
        # Processing the call arguments (line 516)
        int_356009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 27), 'int')
        # Getting the type of 'params' (line 516)
        params_356010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 31), 'params', False)
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'True' (line 516)
        True_356011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 43), 'True', False)
        keyword_356012 = True_356011
        kwargs_356013 = {'sym': keyword_356012}
        # Getting the type of 'window' (line 516)
        window_356008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 20), 'window', False)
        # Calling window(args, kwargs) (line 516)
        window_call_result_356014 = invoke(stypy.reporting.localization.Localization(__file__, 516, 20), window_356008, *[int_356009, params_356010], **kwargs_356013)
        
        # Obtaining the member 'dtype' of a type (line 516)
        dtype_356015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 20), window_call_result_356014, 'dtype')
        str_356016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 58), 'str', 'float')
        # Applying the binary operator '==' (line 516)
        result_eq_356017 = python_operator(stypy.reporting.localization.Localization(__file__, 516, 20), '==', dtype_356015, str_356016)
        
        # Processing the call keyword arguments (line 516)
        kwargs_356018 = {}
        # Getting the type of 'assert_' (line 516)
        assert__356007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 516)
        assert__call_result_356019 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), assert__356007, *[result_eq_356017], **kwargs_356018)
        
        
        # Call to assert_(...): (line 517)
        # Processing the call arguments (line 517)
        
        
        # Call to window(...): (line 517)
        # Processing the call arguments (line 517)
        int_356022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 27), 'int')
        # Getting the type of 'params' (line 517)
        params_356023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 31), 'params', False)
        # Processing the call keyword arguments (line 517)
        # Getting the type of 'False' (line 517)
        False_356024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 43), 'False', False)
        keyword_356025 = False_356024
        kwargs_356026 = {'sym': keyword_356025}
        # Getting the type of 'window' (line 517)
        window_356021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'window', False)
        # Calling window(args, kwargs) (line 517)
        window_call_result_356027 = invoke(stypy.reporting.localization.Localization(__file__, 517, 20), window_356021, *[int_356022, params_356023], **kwargs_356026)
        
        # Obtaining the member 'dtype' of a type (line 517)
        dtype_356028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 20), window_call_result_356027, 'dtype')
        str_356029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 59), 'str', 'float')
        # Applying the binary operator '==' (line 517)
        result_eq_356030 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 20), '==', dtype_356028, str_356029)
        
        # Processing the call keyword arguments (line 517)
        kwargs_356031 = {}
        # Getting the type of 'assert_' (line 517)
        assert__356020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 517)
        assert__call_result_356032 = invoke(stypy.reporting.localization.Localization(__file__, 517, 12), assert__356020, *[result_eq_356030], **kwargs_356031)
        
        
        # Call to assert_(...): (line 518)
        # Processing the call arguments (line 518)
        
        
        # Call to window(...): (line 518)
        # Processing the call arguments (line 518)
        int_356035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 27), 'int')
        # Getting the type of 'params' (line 518)
        params_356036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'params', False)
        # Processing the call keyword arguments (line 518)
        # Getting the type of 'True' (line 518)
        True_356037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 43), 'True', False)
        keyword_356038 = True_356037
        kwargs_356039 = {'sym': keyword_356038}
        # Getting the type of 'window' (line 518)
        window_356034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 20), 'window', False)
        # Calling window(args, kwargs) (line 518)
        window_call_result_356040 = invoke(stypy.reporting.localization.Localization(__file__, 518, 20), window_356034, *[int_356035, params_356036], **kwargs_356039)
        
        # Obtaining the member 'dtype' of a type (line 518)
        dtype_356041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 20), window_call_result_356040, 'dtype')
        str_356042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 58), 'str', 'float')
        # Applying the binary operator '==' (line 518)
        result_eq_356043 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 20), '==', dtype_356041, str_356042)
        
        # Processing the call keyword arguments (line 518)
        kwargs_356044 = {}
        # Getting the type of 'assert_' (line 518)
        assert__356033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 518)
        assert__call_result_356045 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), assert__356033, *[result_eq_356043], **kwargs_356044)
        
        
        # Call to assert_(...): (line 519)
        # Processing the call arguments (line 519)
        
        
        # Call to window(...): (line 519)
        # Processing the call arguments (line 519)
        int_356048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 27), 'int')
        # Getting the type of 'params' (line 519)
        params_356049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 31), 'params', False)
        # Processing the call keyword arguments (line 519)
        # Getting the type of 'False' (line 519)
        False_356050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 43), 'False', False)
        keyword_356051 = False_356050
        kwargs_356052 = {'sym': keyword_356051}
        # Getting the type of 'window' (line 519)
        window_356047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'window', False)
        # Calling window(args, kwargs) (line 519)
        window_call_result_356053 = invoke(stypy.reporting.localization.Localization(__file__, 519, 20), window_356047, *[int_356048, params_356049], **kwargs_356052)
        
        # Obtaining the member 'dtype' of a type (line 519)
        dtype_356054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 20), window_call_result_356053, 'dtype')
        str_356055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 59), 'str', 'float')
        # Applying the binary operator '==' (line 519)
        result_eq_356056 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 20), '==', dtype_356054, str_356055)
        
        # Processing the call keyword arguments (line 519)
        kwargs_356057 = {}
        # Getting the type of 'assert_' (line 519)
        assert__356046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 519)
        assert__call_result_356058 = invoke(stypy.reporting.localization.Localization(__file__, 519, 12), assert__356046, *[result_eq_356056], **kwargs_356057)
        
        
        # Call to assert_array_less(...): (line 522)
        # Processing the call arguments (line 522)
        
        # Call to window(...): (line 522)
        # Processing the call arguments (line 522)
        int_356061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 37), 'int')
        # Getting the type of 'params' (line 522)
        params_356062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 42), 'params', False)
        # Processing the call keyword arguments (line 522)
        # Getting the type of 'True' (line 522)
        True_356063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 54), 'True', False)
        keyword_356064 = True_356063
        kwargs_356065 = {'sym': keyword_356064}
        # Getting the type of 'window' (line 522)
        window_356060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 30), 'window', False)
        # Calling window(args, kwargs) (line 522)
        window_call_result_356066 = invoke(stypy.reporting.localization.Localization(__file__, 522, 30), window_356060, *[int_356061, params_356062], **kwargs_356065)
        
        float_356067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 61), 'float')
        # Processing the call keyword arguments (line 522)
        kwargs_356068 = {}
        # Getting the type of 'assert_array_less' (line 522)
        assert_array_less_356059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 522)
        assert_array_less_call_result_356069 = invoke(stypy.reporting.localization.Localization(__file__, 522, 12), assert_array_less_356059, *[window_call_result_356066, float_356067], **kwargs_356068)
        
        
        # Call to assert_array_less(...): (line 523)
        # Processing the call arguments (line 523)
        
        # Call to window(...): (line 523)
        # Processing the call arguments (line 523)
        int_356072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 37), 'int')
        # Getting the type of 'params' (line 523)
        params_356073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 42), 'params', False)
        # Processing the call keyword arguments (line 523)
        # Getting the type of 'False' (line 523)
        False_356074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 54), 'False', False)
        keyword_356075 = False_356074
        kwargs_356076 = {'sym': keyword_356075}
        # Getting the type of 'window' (line 523)
        window_356071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 30), 'window', False)
        # Calling window(args, kwargs) (line 523)
        window_call_result_356077 = invoke(stypy.reporting.localization.Localization(__file__, 523, 30), window_356071, *[int_356072, params_356073], **kwargs_356076)
        
        float_356078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 62), 'float')
        # Processing the call keyword arguments (line 523)
        kwargs_356079 = {}
        # Getting the type of 'assert_array_less' (line 523)
        assert_array_less_356070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 12), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 523)
        assert_array_less_call_result_356080 = invoke(stypy.reporting.localization.Localization(__file__, 523, 12), assert_array_less_356070, *[window_call_result_356077, float_356078], **kwargs_356079)
        
        
        # Call to assert_array_less(...): (line 524)
        # Processing the call arguments (line 524)
        
        # Call to window(...): (line 524)
        # Processing the call arguments (line 524)
        int_356083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 37), 'int')
        # Getting the type of 'params' (line 524)
        params_356084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'params', False)
        # Processing the call keyword arguments (line 524)
        # Getting the type of 'True' (line 524)
        True_356085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 53), 'True', False)
        keyword_356086 = True_356085
        kwargs_356087 = {'sym': keyword_356086}
        # Getting the type of 'window' (line 524)
        window_356082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 30), 'window', False)
        # Calling window(args, kwargs) (line 524)
        window_call_result_356088 = invoke(stypy.reporting.localization.Localization(__file__, 524, 30), window_356082, *[int_356083, params_356084], **kwargs_356087)
        
        float_356089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 60), 'float')
        # Processing the call keyword arguments (line 524)
        kwargs_356090 = {}
        # Getting the type of 'assert_array_less' (line 524)
        assert_array_less_356081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 12), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 524)
        assert_array_less_call_result_356091 = invoke(stypy.reporting.localization.Localization(__file__, 524, 12), assert_array_less_356081, *[window_call_result_356088, float_356089], **kwargs_356090)
        
        
        # Call to assert_array_less(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Call to window(...): (line 525)
        # Processing the call arguments (line 525)
        int_356094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 37), 'int')
        # Getting the type of 'params' (line 525)
        params_356095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 41), 'params', False)
        # Processing the call keyword arguments (line 525)
        # Getting the type of 'False' (line 525)
        False_356096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 53), 'False', False)
        keyword_356097 = False_356096
        kwargs_356098 = {'sym': keyword_356097}
        # Getting the type of 'window' (line 525)
        window_356093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 30), 'window', False)
        # Calling window(args, kwargs) (line 525)
        window_call_result_356099 = invoke(stypy.reporting.localization.Localization(__file__, 525, 30), window_356093, *[int_356094, params_356095], **kwargs_356098)
        
        float_356100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 61), 'float')
        # Processing the call keyword arguments (line 525)
        kwargs_356101 = {}
        # Getting the type of 'assert_array_less' (line 525)
        assert_array_less_356092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 12), 'assert_array_less', False)
        # Calling assert_array_less(args, kwargs) (line 525)
        assert_array_less_call_result_356102 = invoke(stypy.reporting.localization.Localization(__file__, 525, 12), assert_array_less_356092, *[window_call_result_356099, float_356100], **kwargs_356101)
        
        
        # Call to assert_allclose(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Call to fft(...): (line 528)
        # Processing the call arguments (line 528)
        
        # Call to window(...): (line 528)
        # Processing the call arguments (line 528)
        int_356107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 47), 'int')
        # Getting the type of 'params' (line 528)
        params_356108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 52), 'params', False)
        # Processing the call keyword arguments (line 528)
        # Getting the type of 'False' (line 528)
        False_356109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 64), 'False', False)
        keyword_356110 = False_356109
        kwargs_356111 = {'sym': keyword_356110}
        # Getting the type of 'window' (line 528)
        window_356106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 40), 'window', False)
        # Calling window(args, kwargs) (line 528)
        window_call_result_356112 = invoke(stypy.reporting.localization.Localization(__file__, 528, 40), window_356106, *[int_356107, params_356108], **kwargs_356111)
        
        # Processing the call keyword arguments (line 528)
        kwargs_356113 = {}
        # Getting the type of 'fftpack' (line 528)
        fftpack_356104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 28), 'fftpack', False)
        # Obtaining the member 'fft' of a type (line 528)
        fft_356105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 28), fftpack_356104, 'fft')
        # Calling fft(args, kwargs) (line 528)
        fft_call_result_356114 = invoke(stypy.reporting.localization.Localization(__file__, 528, 28), fft_356105, *[window_call_result_356112], **kwargs_356113)
        
        # Obtaining the member 'imag' of a type (line 528)
        imag_356115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 28), fft_call_result_356114, 'imag')
        int_356116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 28), 'int')
        # Processing the call keyword arguments (line 528)
        float_356117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 36), 'float')
        keyword_356118 = float_356117
        kwargs_356119 = {'atol': keyword_356118}
        # Getting the type of 'assert_allclose' (line 528)
        assert_allclose_356103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 528)
        assert_allclose_call_result_356120 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), assert_allclose_356103, *[imag_356115, int_356116], **kwargs_356119)
        
        
        # Call to assert_allclose(...): (line 530)
        # Processing the call arguments (line 530)
        
        # Call to fft(...): (line 530)
        # Processing the call arguments (line 530)
        
        # Call to window(...): (line 530)
        # Processing the call arguments (line 530)
        int_356125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 47), 'int')
        # Getting the type of 'params' (line 530)
        params_356126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 52), 'params', False)
        # Processing the call keyword arguments (line 530)
        # Getting the type of 'False' (line 530)
        False_356127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 64), 'False', False)
        keyword_356128 = False_356127
        kwargs_356129 = {'sym': keyword_356128}
        # Getting the type of 'window' (line 530)
        window_356124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 40), 'window', False)
        # Calling window(args, kwargs) (line 530)
        window_call_result_356130 = invoke(stypy.reporting.localization.Localization(__file__, 530, 40), window_356124, *[int_356125, params_356126], **kwargs_356129)
        
        # Processing the call keyword arguments (line 530)
        kwargs_356131 = {}
        # Getting the type of 'fftpack' (line 530)
        fftpack_356122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 28), 'fftpack', False)
        # Obtaining the member 'fft' of a type (line 530)
        fft_356123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 28), fftpack_356122, 'fft')
        # Calling fft(args, kwargs) (line 530)
        fft_call_result_356132 = invoke(stypy.reporting.localization.Localization(__file__, 530, 28), fft_356123, *[window_call_result_356130], **kwargs_356131)
        
        # Obtaining the member 'imag' of a type (line 530)
        imag_356133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 28), fft_call_result_356132, 'imag')
        int_356134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 28), 'int')
        # Processing the call keyword arguments (line 530)
        float_356135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 36), 'float')
        keyword_356136 = float_356135
        kwargs_356137 = {'atol': keyword_356136}
        # Getting the type of 'assert_allclose' (line 530)
        assert_allclose_356121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 530)
        assert_allclose_call_result_356138 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), assert_allclose_356121, *[imag_356133, int_356134], **kwargs_356137)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 486)
        exit___356139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 13), suppress_warnings_call_result_355809, '__exit__')
        with_exit_356140 = invoke(stypy.reporting.localization.Localization(__file__, 486, 13), exit___356139, None, None, None)

    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_windowfunc_basics(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_windowfunc_basics' in the type store
    # Getting the type of 'stypy_return_type' (line 483)
    stypy_return_type_356141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_356141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_windowfunc_basics'
    return stypy_return_type_356141

# Assigning a type to the variable 'test_windowfunc_basics' (line 483)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'test_windowfunc_basics', test_windowfunc_basics)

@norecursion
def test_needs_params(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_needs_params'
    module_type_store = module_type_store.open_function_context('test_needs_params', 534, 0, False)
    
    # Passed parameters checking function
    test_needs_params.stypy_localization = localization
    test_needs_params.stypy_type_of_self = None
    test_needs_params.stypy_type_store = module_type_store
    test_needs_params.stypy_function_name = 'test_needs_params'
    test_needs_params.stypy_param_names_list = []
    test_needs_params.stypy_varargs_param_name = None
    test_needs_params.stypy_kwargs_param_name = None
    test_needs_params.stypy_call_defaults = defaults
    test_needs_params.stypy_call_varargs = varargs
    test_needs_params.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_needs_params', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_needs_params', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_needs_params(...)' code ##################

    
    
    # Obtaining an instance of the builtin type 'list' (line 535)
    list_356142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 535)
    # Adding element type (line 535)
    str_356143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 19), 'str', 'kaiser')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356143)
    # Adding element type (line 535)
    str_356144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 29), 'str', 'ksr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356144)
    # Adding element type (line 535)
    str_356145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 36), 'str', 'gaussian')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356145)
    # Adding element type (line 535)
    str_356146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 48), 'str', 'gauss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356146)
    # Adding element type (line 535)
    str_356147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 57), 'str', 'gss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356147)
    # Adding element type (line 535)
    str_356148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 19), 'str', 'general gaussian')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356148)
    # Adding element type (line 535)
    str_356149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 39), 'str', 'general_gaussian')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356149)
    # Adding element type (line 535)
    str_356150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 19), 'str', 'general gauss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356150)
    # Adding element type (line 535)
    str_356151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 36), 'str', 'general_gauss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356151)
    # Adding element type (line 535)
    str_356152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 53), 'str', 'ggs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356152)
    # Adding element type (line 535)
    str_356153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 19), 'str', 'slepian')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356153)
    # Adding element type (line 535)
    str_356154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 30), 'str', 'optimal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356154)
    # Adding element type (line 535)
    str_356155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 41), 'str', 'slep')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356155)
    # Adding element type (line 535)
    str_356156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 49), 'str', 'dss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356156)
    # Adding element type (line 535)
    str_356157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 56), 'str', 'dpss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356157)
    # Adding element type (line 535)
    str_356158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 19), 'str', 'chebwin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356158)
    # Adding element type (line 535)
    str_356159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 30), 'str', 'cheb')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356159)
    # Adding element type (line 535)
    str_356160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 38), 'str', 'exponential')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356160)
    # Adding element type (line 535)
    str_356161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 53), 'str', 'poisson')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356161)
    # Adding element type (line 535)
    str_356162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, 64), 'str', 'tukey')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356162)
    # Adding element type (line 535)
    str_356163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 19), 'str', 'tuk')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 535, 18), list_356142, str_356163)
    
    # Testing the type of a for loop iterable (line 535)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 535, 4), list_356142)
    # Getting the type of the for loop variable (line 535)
    for_loop_var_356164 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 535, 4), list_356142)
    # Assigning a type to the variable 'winstr' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'winstr', for_loop_var_356164)
    # SSA begins for a for statement (line 535)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_raises(...): (line 541)
    # Processing the call arguments (line 541)
    # Getting the type of 'ValueError' (line 541)
    ValueError_356166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 22), 'ValueError', False)
    # Getting the type of 'signal' (line 541)
    signal_356167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 34), 'signal', False)
    # Obtaining the member 'get_window' of a type (line 541)
    get_window_356168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 541, 34), signal_356167, 'get_window')
    # Getting the type of 'winstr' (line 541)
    winstr_356169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 53), 'winstr', False)
    int_356170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 61), 'int')
    # Processing the call keyword arguments (line 541)
    kwargs_356171 = {}
    # Getting the type of 'assert_raises' (line 541)
    assert_raises_356165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 8), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 541)
    assert_raises_call_result_356172 = invoke(stypy.reporting.localization.Localization(__file__, 541, 8), assert_raises_356165, *[ValueError_356166, get_window_356168, winstr_356169, int_356170], **kwargs_356171)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_needs_params(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_needs_params' in the type store
    # Getting the type of 'stypy_return_type' (line 534)
    stypy_return_type_356173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_356173)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_needs_params'
    return stypy_return_type_356173

# Assigning a type to the variable 'test_needs_params' (line 534)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 0), 'test_needs_params', test_needs_params)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
