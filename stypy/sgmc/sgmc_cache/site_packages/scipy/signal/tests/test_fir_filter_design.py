
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
5:                            assert_equal, assert_,
6:                            assert_allclose, assert_warns)
7: from pytest import raises as assert_raises
8: 
9: from scipy.special import sinc
10: 
11: from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
12:         firwin, firwin2, freqz, remez, firls, minimum_phase
13: 
14: 
15: def test_kaiser_beta():
16:     b = kaiser_beta(58.7)
17:     assert_almost_equal(b, 0.1102 * 50.0)
18:     b = kaiser_beta(22.0)
19:     assert_almost_equal(b, 0.5842 + 0.07886)
20:     b = kaiser_beta(21.0)
21:     assert_equal(b, 0.0)
22:     b = kaiser_beta(10.0)
23:     assert_equal(b, 0.0)
24: 
25: 
26: def test_kaiser_atten():
27:     a = kaiser_atten(1, 1.0)
28:     assert_equal(a, 7.95)
29:     a = kaiser_atten(2, 1/np.pi)
30:     assert_equal(a, 2.285 + 7.95)
31: 
32: 
33: def test_kaiserord():
34:     assert_raises(ValueError, kaiserord, 1.0, 1.0)
35:     numtaps, beta = kaiserord(2.285 + 7.95 - 0.001, 1/np.pi)
36:     assert_equal((numtaps, beta), (2, 0.0))
37: 
38: 
39: class TestFirwin(object):
40: 
41:     def check_response(self, h, expected_response, tol=.05):
42:         N = len(h)
43:         alpha = 0.5 * (N-1)
44:         m = np.arange(0,N) - alpha   # time indices of taps
45:         for freq, expected in expected_response:
46:             actual = abs(np.sum(h*np.exp(-1.j*np.pi*m*freq)))
47:             mse = abs(actual-expected)**2
48:             assert_(mse < tol, 'response not as expected, mse=%g > %g'
49:                % (mse, tol))
50: 
51:     def test_response(self):
52:         N = 51
53:         f = .5
54:         # increase length just to try even/odd
55:         h = firwin(N, f)  # low-pass from 0 to f
56:         self.check_response(h, [(.25,1), (.75,0)])
57: 
58:         h = firwin(N+1, f, window='nuttall')  # specific window
59:         self.check_response(h, [(.25,1), (.75,0)])
60: 
61:         h = firwin(N+2, f, pass_zero=False)  # stop from 0 to f --> high-pass
62:         self.check_response(h, [(.25,0), (.75,1)])
63: 
64:         f1, f2, f3, f4 = .2, .4, .6, .8
65:         h = firwin(N+3, [f1, f2], pass_zero=False)  # band-pass filter
66:         self.check_response(h, [(.1,0), (.3,1), (.5,0)])
67: 
68:         h = firwin(N+4, [f1, f2])  # band-stop filter
69:         self.check_response(h, [(.1,1), (.3,0), (.5,1)])
70: 
71:         h = firwin(N+5, [f1, f2, f3, f4], pass_zero=False, scale=False)
72:         self.check_response(h, [(.1,0), (.3,1), (.5,0), (.7,1), (.9,0)])
73: 
74:         h = firwin(N+6, [f1, f2, f3, f4])  # multiband filter
75:         self.check_response(h, [(.1,1), (.3,0), (.5,1), (.7,0), (.9,1)])
76: 
77:         h = firwin(N+7, 0.1, width=.03)  # low-pass
78:         self.check_response(h, [(.05,1), (.75,0)])
79: 
80:         h = firwin(N+8, 0.1, pass_zero=False)  # high-pass
81:         self.check_response(h, [(.05,0), (.75,1)])
82: 
83:     def mse(self, h, bands):
84:         '''Compute mean squared error versus ideal response across frequency
85:         band.
86:           h -- coefficients
87:           bands -- list of (left, right) tuples relative to 1==Nyquist of
88:             passbands
89:         '''
90:         w, H = freqz(h, worN=1024)
91:         f = w/np.pi
92:         passIndicator = np.zeros(len(w), bool)
93:         for left, right in bands:
94:             passIndicator |= (f >= left) & (f < right)
95:         Hideal = np.where(passIndicator, 1, 0)
96:         mse = np.mean(abs(abs(H)-Hideal)**2)
97:         return mse
98: 
99:     def test_scaling(self):
100:         '''
101:         For one lowpass, bandpass, and highpass example filter, this test
102:         checks two things:
103:           - the mean squared error over the frequency domain of the unscaled
104:             filter is smaller than the scaled filter (true for rectangular
105:             window)
106:           - the response of the scaled filter is exactly unity at the center
107:             of the first passband
108:         '''
109:         N = 11
110:         cases = [
111:             ([.5], True, (0, 1)),
112:             ([0.2, .6], False, (.4, 1)),
113:             ([.5], False, (1, 1)),
114:         ]
115:         for cutoff, pass_zero, expected_response in cases:
116:             h = firwin(N, cutoff, scale=False, pass_zero=pass_zero, window='ones')
117:             hs = firwin(N, cutoff, scale=True, pass_zero=pass_zero, window='ones')
118:             if len(cutoff) == 1:
119:                 if pass_zero:
120:                     cutoff = [0] + cutoff
121:                 else:
122:                     cutoff = cutoff + [1]
123:             assert_(self.mse(h, [cutoff]) < self.mse(hs, [cutoff]),
124:                 'least squares violation')
125:             self.check_response(hs, [expected_response], 1e-12)
126: 
127: 
128: class TestFirWinMore(object):
129:     '''Different author, different style, different tests...'''
130: 
131:     def test_lowpass(self):
132:         width = 0.04
133:         ntaps, beta = kaiserord(120, width)
134:         taps = firwin(ntaps, cutoff=0.5, window=('kaiser', beta), scale=False)
135: 
136:         # Check the symmetry of taps.
137:         assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])
138: 
139:         # Check the gain at a few samples where we know it should be approximately 0 or 1.
140:         freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
141:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
142:         assert_array_almost_equal(np.abs(response),
143:                                     [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
144: 
145:     def test_highpass(self):
146:         width = 0.04
147:         ntaps, beta = kaiserord(120, width)
148: 
149:         # Ensure that ntaps is odd.
150:         ntaps |= 1
151: 
152:         taps = firwin(ntaps, cutoff=0.5, window=('kaiser', beta),
153:                         pass_zero=False, scale=False)
154: 
155:         # Check the symmetry of taps.
156:         assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])
157: 
158:         # Check the gain at a few samples where we know it should be approximately 0 or 1.
159:         freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2, 0.75, 1.0])
160:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
161:         assert_array_almost_equal(np.abs(response),
162:                                     [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)
163: 
164:     def test_bandpass(self):
165:         width = 0.04
166:         ntaps, beta = kaiserord(120, width)
167:         taps = firwin(ntaps, cutoff=[0.3, 0.7], window=('kaiser', beta),
168:                         pass_zero=False, scale=False)
169: 
170:         # Check the symmetry of taps.
171:         assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])
172: 
173:         # Check the gain at a few samples where we know it should be approximately 0 or 1.
174:         freq_samples = np.array([0.0, 0.2, 0.3-width/2, 0.3+width/2, 0.5,
175:                                 0.7-width/2, 0.7+width/2, 0.8, 1.0])
176:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
177:         assert_array_almost_equal(np.abs(response),
178:                 [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
179: 
180:     def test_multi(self):
181:         width = 0.04
182:         ntaps, beta = kaiserord(120, width)
183:         taps = firwin(ntaps, cutoff=[0.2, 0.5, 0.8], window=('kaiser', beta),
184:                         pass_zero=True, scale=False)
185: 
186:         # Check the symmetry of taps.
187:         assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])
188: 
189:         # Check the gain at a few samples where we know it should be approximately 0 or 1.
190:         freq_samples = np.array([0.0, 0.1, 0.2-width/2, 0.2+width/2, 0.35,
191:                                 0.5-width/2, 0.5+width/2, 0.65,
192:                                 0.8-width/2, 0.8+width/2, 0.9, 1.0])
193:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
194:         assert_array_almost_equal(np.abs(response),
195:                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
196:                 decimal=5)
197: 
198:     def test_fs_nyq(self):
199:         '''Test the fs and nyq keywords.'''
200:         nyquist = 1000
201:         width = 40.0
202:         relative_width = width/nyquist
203:         ntaps, beta = kaiserord(120, relative_width)
204:         taps = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
205:                         pass_zero=False, scale=False, fs=2*nyquist)
206: 
207:         # Check the symmetry of taps.
208:         assert_array_almost_equal(taps[:ntaps//2], taps[ntaps:ntaps-ntaps//2-1:-1])
209: 
210:         # Check the gain at a few samples where we know it should be approximately 0 or 1.
211:         freq_samples = np.array([0.0, 200, 300-width/2, 300+width/2, 500,
212:                                 700-width/2, 700+width/2, 800, 1000])
213:         freqs, response = freqz(taps, worN=np.pi*freq_samples/nyquist)
214:         assert_array_almost_equal(np.abs(response),
215:                 [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], decimal=5)
216: 
217:         taps2 = firwin(ntaps, cutoff=[300, 700], window=('kaiser', beta),
218:                         pass_zero=False, scale=False, nyq=nyquist)
219:         assert_allclose(taps2, taps)
220: 
221:     def test_bad_cutoff(self):
222:         '''Test that invalid cutoff argument raises ValueError.'''
223:         # cutoff values must be greater than 0 and less than 1.
224:         assert_raises(ValueError, firwin, 99, -0.5)
225:         assert_raises(ValueError, firwin, 99, 1.5)
226:         # Don't allow 0 or 1 in cutoff.
227:         assert_raises(ValueError, firwin, 99, [0, 0.5])
228:         assert_raises(ValueError, firwin, 99, [0.5, 1])
229:         # cutoff values must be strictly increasing.
230:         assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.2])
231:         assert_raises(ValueError, firwin, 99, [0.1, 0.5, 0.5])
232:         # Must have at least one cutoff value.
233:         assert_raises(ValueError, firwin, 99, [])
234:         # 2D array not allowed.
235:         assert_raises(ValueError, firwin, 99, [[0.1, 0.2],[0.3, 0.4]])
236:         # cutoff values must be less than nyq.
237:         assert_raises(ValueError, firwin, 99, 50.0, nyq=40)
238:         assert_raises(ValueError, firwin, 99, [10, 20, 30], nyq=25)
239:         assert_raises(ValueError, firwin, 99, 50.0, fs=80)
240:         assert_raises(ValueError, firwin, 99, [10, 20, 30], fs=50)
241: 
242:     def test_even_highpass_raises_value_error(self):
243:         '''Test that attempt to create a highpass filter with an even number
244:         of taps raises a ValueError exception.'''
245:         assert_raises(ValueError, firwin, 40, 0.5, pass_zero=False)
246:         assert_raises(ValueError, firwin, 40, [.25, 0.5])
247: 
248: 
249: class TestFirwin2(object):
250: 
251:     def test_invalid_args(self):
252:         # `freq` and `gain` have different lengths.
253:         assert_raises(ValueError, firwin2, 50, [0, 0.5, 1], [0.0, 1.0])
254:         # `nfreqs` is less than `ntaps`.
255:         assert_raises(ValueError, firwin2, 50, [0, 0.5, 1], [0.0, 1.0, 1.0], nfreqs=33)
256:         # Decreasing value in `freq`
257:         assert_raises(ValueError, firwin2, 50, [0, 0.5, 0.4, 1.0], [0, .25, .5, 1.0])
258:         # Value in `freq` repeated more than once.
259:         assert_raises(ValueError, firwin2, 50, [0, .1, .1, .1, 1.0],
260:                                                [0.0, 0.5, 0.75, 1.0, 1.0])
261:         # `freq` does not start at 0.0.
262:         assert_raises(ValueError, firwin2, 50, [0.5, 1.0], [0.0, 1.0])
263: 
264:         # Type II filter, but the gain at nyquist rate is not zero.
265:         assert_raises(ValueError, firwin2, 16, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
266: 
267:         # Type III filter, but the gains at nyquist and zero rate are not zero.
268:         assert_raises(ValueError, firwin2, 17, [0.0, 0.5, 1.0], [0.0, 1.0, 1.0],
269:                       antisymmetric=True)
270:         assert_raises(ValueError, firwin2, 17, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0],
271:                       antisymmetric=True)
272:         assert_raises(ValueError, firwin2, 17, [0.0, 0.5, 1.0], [1.0, 1.0, 1.0],
273:                       antisymmetric=True)
274: 
275:         # Type VI filter, but the gain at zero rate is not zero.
276:         assert_raises(ValueError, firwin2, 16, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0],
277:                       antisymmetric=True)
278: 
279:     def test01(self):
280:         width = 0.04
281:         beta = 12.0
282:         ntaps = 400
283:         # Filter is 1 from w=0 to w=0.5, then decreases linearly from 1 to 0 as w
284:         # increases from w=0.5 to w=1  (w=1 is the Nyquist frequency).
285:         freq = [0.0, 0.5, 1.0]
286:         gain = [1.0, 1.0, 0.0]
287:         taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
288:         freq_samples = np.array([0.0, 0.25, 0.5-width/2, 0.5+width/2,
289:                                                         0.75, 1.0-width/2])
290:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
291:         assert_array_almost_equal(np.abs(response),
292:                         [1.0, 1.0, 1.0, 1.0-width, 0.5, width], decimal=5)
293: 
294:     def test02(self):
295:         width = 0.04
296:         beta = 12.0
297:         # ntaps must be odd for positive gain at Nyquist.
298:         ntaps = 401
299:         # An ideal highpass filter.
300:         freq = [0.0, 0.5, 0.5, 1.0]
301:         gain = [0.0, 0.0, 1.0, 1.0]
302:         taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
303:         freq_samples = np.array([0.0, 0.25, 0.5-width, 0.5+width, 0.75, 1.0])
304:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
305:         assert_array_almost_equal(np.abs(response),
306:                                 [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)
307: 
308:     def test03(self):
309:         width = 0.02
310:         ntaps, beta = kaiserord(120, width)
311:         # ntaps must be odd for positive gain at Nyquist.
312:         ntaps = int(ntaps) | 1
313:         freq = [0.0, 0.4, 0.4, 0.5, 0.5, 1.0]
314:         gain = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
315:         taps = firwin2(ntaps, freq, gain, window=('kaiser', beta))
316:         freq_samples = np.array([0.0, 0.4-width, 0.4+width, 0.45,
317:                                     0.5-width, 0.5+width, 0.75, 1.0])
318:         freqs, response = freqz(taps, worN=np.pi*freq_samples)
319:         assert_array_almost_equal(np.abs(response),
320:                     [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], decimal=5)
321: 
322:     def test04(self):
323:         '''Test firwin2 when window=None.'''
324:         ntaps = 5
325:         # Ideal lowpass: gain is 1 on [0,0.5], and 0 on [0.5, 1.0]
326:         freq = [0.0, 0.5, 0.5, 1.0]
327:         gain = [1.0, 1.0, 0.0, 0.0]
328:         taps = firwin2(ntaps, freq, gain, window=None, nfreqs=8193)
329:         alpha = 0.5 * (ntaps - 1)
330:         m = np.arange(0, ntaps) - alpha
331:         h = 0.5 * sinc(0.5 * m)
332:         assert_array_almost_equal(h, taps)
333: 
334:     def test05(self):
335:         '''Test firwin2 for calculating Type IV filters'''
336:         ntaps = 1500
337: 
338:         freq = [0.0, 1.0]
339:         gain = [0.0, 1.0]
340:         taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
341:         assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2:][::-1])
342: 
343:         freqs, response = freqz(taps, worN=2048)
344:         assert_array_almost_equal(abs(response), freqs / np.pi, decimal=4)
345: 
346:     def test06(self):
347:         '''Test firwin2 for calculating Type III filters'''
348:         ntaps = 1501
349: 
350:         freq = [0.0, 0.5, 0.55, 1.0]
351:         gain = [0.0, 0.5, 0.0, 0.0]
352:         taps = firwin2(ntaps, freq, gain, window=None, antisymmetric=True)
353:         assert_equal(taps[ntaps // 2], 0.0)
354:         assert_array_almost_equal(taps[: ntaps // 2], -taps[ntaps // 2 + 1:][::-1])
355: 
356:         freqs, response1 = freqz(taps, worN=2048)
357:         response2 = np.interp(freqs / np.pi, freq, gain)
358:         assert_array_almost_equal(abs(response1), response2, decimal=3)
359: 
360:     def test_fs_nyq(self):
361:         taps1 = firwin2(80, [0.0, 0.5, 1.0], [1.0, 1.0, 0.0])
362:         taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], fs=120.0)
363:         assert_array_almost_equal(taps1, taps2)
364:         taps2 = firwin2(80, [0.0, 30.0, 60.0], [1.0, 1.0, 0.0], nyq=60.0)
365:         assert_array_almost_equal(taps1, taps2)
366: 
367: class TestRemez(object):
368: 
369:     def test_bad_args(self):
370:         assert_raises(ValueError, remez, 11, [0.1, 0.4], [1], type='pooka')
371: 
372:     def test_hilbert(self):
373:         N = 11  # number of taps in the filter
374:         a = 0.1  # width of the transition band
375: 
376:         # design an unity gain hilbert bandpass filter from w to 0.5-w
377:         h = remez(11, [a, 0.5-a], [1], type='hilbert')
378: 
379:         # make sure the filter has correct # of taps
380:         assert_(len(h) == N, "Number of Taps")
381: 
382:         # make sure it is type III (anti-symmetric tap coefficients)
383:         assert_array_almost_equal(h[:(N-1)//2], -h[:-(N-1)//2-1:-1])
384: 
385:         # Since the requested response is symmetric, all even coeffcients
386:         # should be zero (or in this case really small)
387:         assert_((abs(h[1::2]) < 1e-15).all(), "Even Coefficients Equal Zero")
388: 
389:         # now check the frequency response
390:         w, H = freqz(h, 1)
391:         f = w/2/np.pi
392:         Hmag = abs(H)
393: 
394:         # should have a zero at 0 and pi (in this case close to zero)
395:         assert_((Hmag[[0, -1]] < 0.02).all(), "Zero at zero and pi")
396: 
397:         # check that the pass band is close to unity
398:         idx = np.logical_and(f > a, f < 0.5-a)
399:         assert_((abs(Hmag[idx] - 1) < 0.015).all(), "Pass Band Close To Unity")
400: 
401:     def test_compare(self):
402:         # test comparison to MATLAB
403:         k = [0.024590270518440, -0.041314581814658, -0.075943803756711,
404:              -0.003530911231040, 0.193140296954975, 0.373400753484939,
405:              0.373400753484939, 0.193140296954975, -0.003530911231040,
406:              -0.075943803756711, -0.041314581814658, 0.024590270518440]
407:         h = remez(12, [0, 0.3, 0.5, 1], [1, 0], Hz=2.)
408:         assert_allclose(h, k)
409:         h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
410:         assert_allclose(h, k)
411: 
412:         h = [-0.038976016082299, 0.018704846485491, -0.014644062687875,
413:              0.002879152556419, 0.016849978528150, -0.043276706138248,
414:              0.073641298245579, -0.103908158578635, 0.129770906801075,
415:              -0.147163447297124, 0.153302248456347, -0.147163447297124,
416:              0.129770906801075, -0.103908158578635, 0.073641298245579,
417:              -0.043276706138248, 0.016849978528150, 0.002879152556419,
418:              -0.014644062687875, 0.018704846485491, -0.038976016082299]
419:         assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], Hz=2.), h)
420:         assert_allclose(remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.), h)
421: 
422: 
423: class TestFirls(object):
424: 
425:     def test_bad_args(self):
426:         # even numtaps
427:         assert_raises(ValueError, firls, 10, [0.1, 0.2], [0, 0])
428:         # odd bands
429:         assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.4], [0, 0, 0])
430:         # len(bands) != len(desired)
431:         assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.4], [0, 0, 0])
432:         # non-monotonic bands
433:         assert_raises(ValueError, firls, 11, [0.2, 0.1], [0, 0])
434:         assert_raises(ValueError, firls, 11, [0.1, 0.2, 0.3, 0.3], [0] * 4)
435:         assert_raises(ValueError, firls, 11, [0.3, 0.4, 0.1, 0.2], [0] * 4)
436:         assert_raises(ValueError, firls, 11, [0.1, 0.3, 0.2, 0.4], [0] * 4)
437:         # negative desired
438:         assert_raises(ValueError, firls, 11, [0.1, 0.2], [-1, 1])
439:         # len(weight) != len(pairs)
440:         assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [1, 2])
441:         # negative weight
442:         assert_raises(ValueError, firls, 11, [0.1, 0.2], [0, 0], [-1])
443: 
444:     def test_firls(self):
445:         N = 11  # number of taps in the filter
446:         a = 0.1  # width of the transition band
447: 
448:         # design a halfband symmetric low-pass filter
449:         h = firls(11, [0, a, 0.5-a, 0.5], [1, 1, 0, 0], fs=1.0)
450: 
451:         # make sure the filter has correct # of taps
452:         assert_equal(len(h), N)
453: 
454:         # make sure it is symmetric
455:         midx = (N-1) // 2
456:         assert_array_almost_equal(h[:midx], h[:-midx-1:-1])
457: 
458:         # make sure the center tap is 0.5
459:         assert_almost_equal(h[midx], 0.5)
460: 
461:         # For halfband symmetric, odd coefficients (except the center)
462:         # should be zero (really small)
463:         hodd = np.hstack((h[1:midx:2], h[-midx+1::2]))
464:         assert_array_almost_equal(hodd, 0)
465: 
466:         # now check the frequency response
467:         w, H = freqz(h, 1)
468:         f = w/2/np.pi
469:         Hmag = np.abs(H)
470: 
471:         # check that the pass band is close to unity
472:         idx = np.logical_and(f > 0, f < a)
473:         assert_array_almost_equal(Hmag[idx], 1, decimal=3)
474: 
475:         # check that the stop band is close to zero
476:         idx = np.logical_and(f > 0.5-a, f < 0.5)
477:         assert_array_almost_equal(Hmag[idx], 0, decimal=3)
478: 
479:     def test_compare(self):
480:         # compare to OCTAVE output
481:         taps = firls(9, [0, 0.5, 0.55, 1], [1, 1, 0, 0], [1, 2])
482:         # >> taps = firls(8, [0 0.5 0.55 1], [1 1 0 0], [1, 2]);
483:         known_taps = [-6.26930101730182e-04, -1.03354450635036e-01,
484:                       -9.81576747564301e-03, 3.17271686090449e-01,
485:                       5.11409425599933e-01, 3.17271686090449e-01,
486:                       -9.81576747564301e-03, -1.03354450635036e-01,
487:                       -6.26930101730182e-04]
488:         assert_allclose(taps, known_taps)
489: 
490:         # compare to MATLAB output
491:         taps = firls(11, [0, 0.5, 0.5, 1], [1, 1, 0, 0], [1, 2])
492:         # >> taps = firls(10, [0 0.5 0.5 1], [1 1 0 0], [1, 2]);
493:         known_taps = [
494:             0.058545300496815, -0.014233383714318, -0.104688258464392,
495:             0.012403323025279, 0.317930861136062, 0.488047220029700,
496:             0.317930861136062, 0.012403323025279, -0.104688258464392,
497:             -0.014233383714318, 0.058545300496815]
498:         assert_allclose(taps, known_taps)
499: 
500:         # With linear changes:
501:         taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], fs=20)
502:         # >> taps = firls(6, [0, 0.1, 0.2, 0.3, 0.4, 0.5], [1, 0, 0, 1, 1, 0])
503:         known_taps = [
504:             1.156090832768218, -4.1385894727395849, 7.5288619164321826,
505:             -8.5530572592947856, 7.5288619164321826, -4.1385894727395849,
506:             1.156090832768218]
507:         assert_allclose(taps, known_taps)
508: 
509:         taps = firls(7, (0, 1, 2, 3, 4, 5), [1, 0, 0, 1, 1, 0], nyq=10)
510:         assert_allclose(taps, known_taps)
511: 
512: 
513: class TestMinimumPhase(object):
514: 
515:     def test_bad_args(self):
516:         # not enough taps
517:         assert_raises(ValueError, minimum_phase, [1.])
518:         assert_raises(ValueError, minimum_phase, [1., 1.])
519:         assert_raises(ValueError, minimum_phase, np.ones(10) * 1j)
520:         assert_raises(ValueError, minimum_phase, 'foo')
521:         assert_raises(ValueError, minimum_phase, np.ones(10), n_fft=8)
522:         assert_raises(ValueError, minimum_phase, np.ones(10), method='foo')
523:         assert_warns(RuntimeWarning, minimum_phase, np.arange(3))
524: 
525:     def test_homomorphic(self):
526:         # check that it can recover frequency responses of arbitrary
527:         # linear-phase filters
528: 
529:         # for some cases we can get the actual filter back
530:         h = [1, -1]
531:         h_new = minimum_phase(np.convolve(h, h[::-1]))
532:         assert_allclose(h_new, h, rtol=0.05)
533: 
534:         # but in general we only guarantee we get the magnitude back
535:         rng = np.random.RandomState(0)
536:         for n in (2, 3, 10, 11, 15, 16, 17, 20, 21, 100, 101):
537:             h = rng.randn(n)
538:             h_new = minimum_phase(np.convolve(h, h[::-1]))
539:             assert_allclose(np.abs(np.fft.fft(h_new)),
540:                             np.abs(np.fft.fft(h)), rtol=1e-4)
541: 
542:     def test_hilbert(self):
543:         # compare to MATLAB output of reference implementation
544: 
545:         # f=[0 0.3 0.5 1];
546:         # a=[1 1 0 0];
547:         # h=remez(11,f,a);
548:         h = remez(12, [0, 0.3, 0.5, 1], [1, 0], fs=2.)
549:         k = [0.349585548646686, 0.373552164395447, 0.326082685363438,
550:              0.077152207480935, -0.129943946349364, -0.059355880509749]
551:         m = minimum_phase(h, 'hilbert')
552:         assert_allclose(m, k, rtol=1e-3)
553: 
554:         # f=[0 0.8 0.9 1];
555:         # a=[0 0 1 1];
556:         # h=remez(20,f,a);
557:         h = remez(21, [0, 0.8, 0.9, 1], [0, 1], fs=2.)
558:         k = [0.232486803906329, -0.133551833687071, 0.151871456867244,
559:              -0.157957283165866, 0.151739294892963, -0.129293146705090,
560:              0.100787844523204, -0.065832656741252, 0.035361328741024,
561:              -0.014977068692269, -0.158416139047557]
562:         m = minimum_phase(h, 'hilbert', n_fft=2**19)
563:         assert_allclose(m, k, rtol=1e-3)
564: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_313945 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_313945) is not StypyTypeError):

    if (import_313945 != 'pyd_module'):
        __import__(import_313945)
        sys_modules_313946 = sys.modules[import_313945]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_313946.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_313945)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal, assert_, assert_allclose, assert_warns' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_313947 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_313947) is not StypyTypeError):

    if (import_313947 != 'pyd_module'):
        __import__(import_313947)
        sys_modules_313948 = sys.modules[import_313947]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_313948.module_type_store, module_type_store, ['assert_almost_equal', 'assert_array_almost_equal', 'assert_equal', 'assert_', 'assert_allclose', 'assert_warns'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_313948, sys_modules_313948.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal, assert_, assert_allclose, assert_warns

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_array_almost_equal', 'assert_equal', 'assert_', 'assert_allclose', 'assert_warns'], [assert_almost_equal, assert_array_almost_equal, assert_equal, assert_, assert_allclose, assert_warns])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_313947)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_313949 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_313949) is not StypyTypeError):

    if (import_313949 != 'pyd_module'):
        __import__(import_313949)
        sys_modules_313950 = sys.modules[import_313949]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_313950.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_313950, sys_modules_313950.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_313949)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.special import sinc' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_313951 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special')

if (type(import_313951) is not StypyTypeError):

    if (import_313951 != 'pyd_module'):
        __import__(import_313951)
        sys_modules_313952 = sys.modules[import_313951]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special', sys_modules_313952.module_type_store, module_type_store, ['sinc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_313952, sys_modules_313952.module_type_store, module_type_store)
    else:
        from scipy.special import sinc

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special', None, module_type_store, ['sinc'], [sinc])

else:
    # Assigning a type to the variable 'scipy.special' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.special', import_313951)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, firwin, firwin2, freqz, remez, firls, minimum_phase' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_313953 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal')

if (type(import_313953) is not StypyTypeError):

    if (import_313953 != 'pyd_module'):
        __import__(import_313953)
        sys_modules_313954 = sys.modules[import_313953]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal', sys_modules_313954.module_type_store, module_type_store, ['kaiser_beta', 'kaiser_atten', 'kaiserord', 'firwin', 'firwin2', 'freqz', 'remez', 'firls', 'minimum_phase'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_313954, sys_modules_313954.module_type_store, module_type_store)
    else:
        from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, firwin, firwin2, freqz, remez, firls, minimum_phase

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal', None, module_type_store, ['kaiser_beta', 'kaiser_atten', 'kaiserord', 'firwin', 'firwin2', 'freqz', 'remez', 'firls', 'minimum_phase'], [kaiser_beta, kaiser_atten, kaiserord, firwin, firwin2, freqz, remez, firls, minimum_phase])

else:
    # Assigning a type to the variable 'scipy.signal' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.signal', import_313953)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


@norecursion
def test_kaiser_beta(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kaiser_beta'
    module_type_store = module_type_store.open_function_context('test_kaiser_beta', 15, 0, False)
    
    # Passed parameters checking function
    test_kaiser_beta.stypy_localization = localization
    test_kaiser_beta.stypy_type_of_self = None
    test_kaiser_beta.stypy_type_store = module_type_store
    test_kaiser_beta.stypy_function_name = 'test_kaiser_beta'
    test_kaiser_beta.stypy_param_names_list = []
    test_kaiser_beta.stypy_varargs_param_name = None
    test_kaiser_beta.stypy_kwargs_param_name = None
    test_kaiser_beta.stypy_call_defaults = defaults
    test_kaiser_beta.stypy_call_varargs = varargs
    test_kaiser_beta.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kaiser_beta', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kaiser_beta', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kaiser_beta(...)' code ##################

    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to kaiser_beta(...): (line 16)
    # Processing the call arguments (line 16)
    float_313956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'float')
    # Processing the call keyword arguments (line 16)
    kwargs_313957 = {}
    # Getting the type of 'kaiser_beta' (line 16)
    kaiser_beta_313955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'kaiser_beta', False)
    # Calling kaiser_beta(args, kwargs) (line 16)
    kaiser_beta_call_result_313958 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), kaiser_beta_313955, *[float_313956], **kwargs_313957)
    
    # Assigning a type to the variable 'b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'b', kaiser_beta_call_result_313958)
    
    # Call to assert_almost_equal(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'b' (line 17)
    b_313960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'b', False)
    float_313961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'float')
    float_313962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'float')
    # Applying the binary operator '*' (line 17)
    result_mul_313963 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 27), '*', float_313961, float_313962)
    
    # Processing the call keyword arguments (line 17)
    kwargs_313964 = {}
    # Getting the type of 'assert_almost_equal' (line 17)
    assert_almost_equal_313959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 17)
    assert_almost_equal_call_result_313965 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_almost_equal_313959, *[b_313960, result_mul_313963], **kwargs_313964)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to kaiser_beta(...): (line 18)
    # Processing the call arguments (line 18)
    float_313967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'float')
    # Processing the call keyword arguments (line 18)
    kwargs_313968 = {}
    # Getting the type of 'kaiser_beta' (line 18)
    kaiser_beta_313966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'kaiser_beta', False)
    # Calling kaiser_beta(args, kwargs) (line 18)
    kaiser_beta_call_result_313969 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), kaiser_beta_313966, *[float_313967], **kwargs_313968)
    
    # Assigning a type to the variable 'b' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'b', kaiser_beta_call_result_313969)
    
    # Call to assert_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'b' (line 19)
    b_313971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'b', False)
    float_313972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'float')
    float_313973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'float')
    # Applying the binary operator '+' (line 19)
    result_add_313974 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 27), '+', float_313972, float_313973)
    
    # Processing the call keyword arguments (line 19)
    kwargs_313975 = {}
    # Getting the type of 'assert_almost_equal' (line 19)
    assert_almost_equal_313970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 19)
    assert_almost_equal_call_result_313976 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_almost_equal_313970, *[b_313971, result_add_313974], **kwargs_313975)
    
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to kaiser_beta(...): (line 20)
    # Processing the call arguments (line 20)
    float_313978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'float')
    # Processing the call keyword arguments (line 20)
    kwargs_313979 = {}
    # Getting the type of 'kaiser_beta' (line 20)
    kaiser_beta_313977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'kaiser_beta', False)
    # Calling kaiser_beta(args, kwargs) (line 20)
    kaiser_beta_call_result_313980 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), kaiser_beta_313977, *[float_313978], **kwargs_313979)
    
    # Assigning a type to the variable 'b' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'b', kaiser_beta_call_result_313980)
    
    # Call to assert_equal(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'b' (line 21)
    b_313982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'b', False)
    float_313983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'float')
    # Processing the call keyword arguments (line 21)
    kwargs_313984 = {}
    # Getting the type of 'assert_equal' (line 21)
    assert_equal_313981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 21)
    assert_equal_call_result_313985 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert_equal_313981, *[b_313982, float_313983], **kwargs_313984)
    
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to kaiser_beta(...): (line 22)
    # Processing the call arguments (line 22)
    float_313987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'float')
    # Processing the call keyword arguments (line 22)
    kwargs_313988 = {}
    # Getting the type of 'kaiser_beta' (line 22)
    kaiser_beta_313986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'kaiser_beta', False)
    # Calling kaiser_beta(args, kwargs) (line 22)
    kaiser_beta_call_result_313989 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), kaiser_beta_313986, *[float_313987], **kwargs_313988)
    
    # Assigning a type to the variable 'b' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'b', kaiser_beta_call_result_313989)
    
    # Call to assert_equal(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'b' (line 23)
    b_313991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'b', False)
    float_313992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'float')
    # Processing the call keyword arguments (line 23)
    kwargs_313993 = {}
    # Getting the type of 'assert_equal' (line 23)
    assert_equal_313990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 23)
    assert_equal_call_result_313994 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_equal_313990, *[b_313991, float_313992], **kwargs_313993)
    
    
    # ################# End of 'test_kaiser_beta(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kaiser_beta' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_313995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_313995)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kaiser_beta'
    return stypy_return_type_313995

# Assigning a type to the variable 'test_kaiser_beta' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_kaiser_beta', test_kaiser_beta)

@norecursion
def test_kaiser_atten(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kaiser_atten'
    module_type_store = module_type_store.open_function_context('test_kaiser_atten', 26, 0, False)
    
    # Passed parameters checking function
    test_kaiser_atten.stypy_localization = localization
    test_kaiser_atten.stypy_type_of_self = None
    test_kaiser_atten.stypy_type_store = module_type_store
    test_kaiser_atten.stypy_function_name = 'test_kaiser_atten'
    test_kaiser_atten.stypy_param_names_list = []
    test_kaiser_atten.stypy_varargs_param_name = None
    test_kaiser_atten.stypy_kwargs_param_name = None
    test_kaiser_atten.stypy_call_defaults = defaults
    test_kaiser_atten.stypy_call_varargs = varargs
    test_kaiser_atten.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kaiser_atten', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kaiser_atten', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kaiser_atten(...)' code ##################

    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to kaiser_atten(...): (line 27)
    # Processing the call arguments (line 27)
    int_313997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'int')
    float_313998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'float')
    # Processing the call keyword arguments (line 27)
    kwargs_313999 = {}
    # Getting the type of 'kaiser_atten' (line 27)
    kaiser_atten_313996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'kaiser_atten', False)
    # Calling kaiser_atten(args, kwargs) (line 27)
    kaiser_atten_call_result_314000 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), kaiser_atten_313996, *[int_313997, float_313998], **kwargs_313999)
    
    # Assigning a type to the variable 'a' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'a', kaiser_atten_call_result_314000)
    
    # Call to assert_equal(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'a' (line 28)
    a_314002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'a', False)
    float_314003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'float')
    # Processing the call keyword arguments (line 28)
    kwargs_314004 = {}
    # Getting the type of 'assert_equal' (line 28)
    assert_equal_314001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 28)
    assert_equal_call_result_314005 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), assert_equal_314001, *[a_314002, float_314003], **kwargs_314004)
    
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to kaiser_atten(...): (line 29)
    # Processing the call arguments (line 29)
    int_314007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'int')
    int_314008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'int')
    # Getting the type of 'np' (line 29)
    np_314009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'np', False)
    # Obtaining the member 'pi' of a type (line 29)
    pi_314010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), np_314009, 'pi')
    # Applying the binary operator 'div' (line 29)
    result_div_314011 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 24), 'div', int_314008, pi_314010)
    
    # Processing the call keyword arguments (line 29)
    kwargs_314012 = {}
    # Getting the type of 'kaiser_atten' (line 29)
    kaiser_atten_314006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'kaiser_atten', False)
    # Calling kaiser_atten(args, kwargs) (line 29)
    kaiser_atten_call_result_314013 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), kaiser_atten_314006, *[int_314007, result_div_314011], **kwargs_314012)
    
    # Assigning a type to the variable 'a' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'a', kaiser_atten_call_result_314013)
    
    # Call to assert_equal(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'a' (line 30)
    a_314015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'a', False)
    float_314016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'float')
    float_314017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'float')
    # Applying the binary operator '+' (line 30)
    result_add_314018 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 20), '+', float_314016, float_314017)
    
    # Processing the call keyword arguments (line 30)
    kwargs_314019 = {}
    # Getting the type of 'assert_equal' (line 30)
    assert_equal_314014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 30)
    assert_equal_call_result_314020 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_equal_314014, *[a_314015, result_add_314018], **kwargs_314019)
    
    
    # ################# End of 'test_kaiser_atten(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kaiser_atten' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_314021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_314021)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kaiser_atten'
    return stypy_return_type_314021

# Assigning a type to the variable 'test_kaiser_atten' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'test_kaiser_atten', test_kaiser_atten)

@norecursion
def test_kaiserord(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_kaiserord'
    module_type_store = module_type_store.open_function_context('test_kaiserord', 33, 0, False)
    
    # Passed parameters checking function
    test_kaiserord.stypy_localization = localization
    test_kaiserord.stypy_type_of_self = None
    test_kaiserord.stypy_type_store = module_type_store
    test_kaiserord.stypy_function_name = 'test_kaiserord'
    test_kaiserord.stypy_param_names_list = []
    test_kaiserord.stypy_varargs_param_name = None
    test_kaiserord.stypy_kwargs_param_name = None
    test_kaiserord.stypy_call_defaults = defaults
    test_kaiserord.stypy_call_varargs = varargs
    test_kaiserord.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_kaiserord', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_kaiserord', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_kaiserord(...)' code ##################

    
    # Call to assert_raises(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'ValueError' (line 34)
    ValueError_314023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'ValueError', False)
    # Getting the type of 'kaiserord' (line 34)
    kaiserord_314024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'kaiserord', False)
    float_314025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'float')
    float_314026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'float')
    # Processing the call keyword arguments (line 34)
    kwargs_314027 = {}
    # Getting the type of 'assert_raises' (line 34)
    assert_raises_314022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 34)
    assert_raises_call_result_314028 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert_raises_314022, *[ValueError_314023, kaiserord_314024, float_314025, float_314026], **kwargs_314027)
    
    
    # Assigning a Call to a Tuple (line 35):
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    int_314029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
    
    # Call to kaiserord(...): (line 35)
    # Processing the call arguments (line 35)
    float_314031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'float')
    float_314032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'float')
    # Applying the binary operator '+' (line 35)
    result_add_314033 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 30), '+', float_314031, float_314032)
    
    float_314034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 45), 'float')
    # Applying the binary operator '-' (line 35)
    result_sub_314035 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 43), '-', result_add_314033, float_314034)
    
    int_314036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 52), 'int')
    # Getting the type of 'np' (line 35)
    np_314037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 54), 'np', False)
    # Obtaining the member 'pi' of a type (line 35)
    pi_314038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 54), np_314037, 'pi')
    # Applying the binary operator 'div' (line 35)
    result_div_314039 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 52), 'div', int_314036, pi_314038)
    
    # Processing the call keyword arguments (line 35)
    kwargs_314040 = {}
    # Getting the type of 'kaiserord' (line 35)
    kaiserord_314030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'kaiserord', False)
    # Calling kaiserord(args, kwargs) (line 35)
    kaiserord_call_result_314041 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), kaiserord_314030, *[result_sub_314035, result_div_314039], **kwargs_314040)
    
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___314042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), kaiserord_call_result_314041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_314043 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), getitem___314042, int_314029)
    
    # Assigning a type to the variable 'tuple_var_assignment_313901' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_313901', subscript_call_result_314043)
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    int_314044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'int')
    
    # Call to kaiserord(...): (line 35)
    # Processing the call arguments (line 35)
    float_314046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'float')
    float_314047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'float')
    # Applying the binary operator '+' (line 35)
    result_add_314048 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 30), '+', float_314046, float_314047)
    
    float_314049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 45), 'float')
    # Applying the binary operator '-' (line 35)
    result_sub_314050 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 43), '-', result_add_314048, float_314049)
    
    int_314051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 52), 'int')
    # Getting the type of 'np' (line 35)
    np_314052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 54), 'np', False)
    # Obtaining the member 'pi' of a type (line 35)
    pi_314053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 54), np_314052, 'pi')
    # Applying the binary operator 'div' (line 35)
    result_div_314054 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 52), 'div', int_314051, pi_314053)
    
    # Processing the call keyword arguments (line 35)
    kwargs_314055 = {}
    # Getting the type of 'kaiserord' (line 35)
    kaiserord_314045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'kaiserord', False)
    # Calling kaiserord(args, kwargs) (line 35)
    kaiserord_call_result_314056 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), kaiserord_314045, *[result_sub_314050, result_div_314054], **kwargs_314055)
    
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___314057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), kaiserord_call_result_314056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_314058 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), getitem___314057, int_314044)
    
    # Assigning a type to the variable 'tuple_var_assignment_313902' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_313902', subscript_call_result_314058)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'tuple_var_assignment_313901' (line 35)
    tuple_var_assignment_313901_314059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_313901')
    # Assigning a type to the variable 'numtaps' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'numtaps', tuple_var_assignment_313901_314059)
    
    # Assigning a Name to a Name (line 35):
    # Getting the type of 'tuple_var_assignment_313902' (line 35)
    tuple_var_assignment_313902_314060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'tuple_var_assignment_313902')
    # Assigning a type to the variable 'beta' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'beta', tuple_var_assignment_313902_314060)
    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_314062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'numtaps' (line 36)
    numtaps_314063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'numtaps', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), tuple_314062, numtaps_314063)
    # Adding element type (line 36)
    # Getting the type of 'beta' (line 36)
    beta_314064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'beta', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), tuple_314062, beta_314064)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_314065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    int_314066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 35), tuple_314065, int_314066)
    # Adding element type (line 36)
    float_314067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 35), tuple_314065, float_314067)
    
    # Processing the call keyword arguments (line 36)
    kwargs_314068 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_314061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_314069 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_equal_314061, *[tuple_314062, tuple_314065], **kwargs_314068)
    
    
    # ################# End of 'test_kaiserord(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_kaiserord' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_314070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_314070)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_kaiserord'
    return stypy_return_type_314070

# Assigning a type to the variable 'test_kaiserord' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'test_kaiserord', test_kaiserord)
# Declaration of the 'TestFirwin' class

class TestFirwin(object, ):

    @norecursion
    def check_response(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_314071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 55), 'float')
        defaults = [float_314071]
        # Create a new context for function 'check_response'
        module_type_store = module_type_store.open_function_context('check_response', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin.check_response.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin.check_response.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin.check_response.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin.check_response.__dict__.__setitem__('stypy_function_name', 'TestFirwin.check_response')
        TestFirwin.check_response.__dict__.__setitem__('stypy_param_names_list', ['h', 'expected_response', 'tol'])
        TestFirwin.check_response.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin.check_response.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin.check_response.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin.check_response.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin.check_response.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin.check_response.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin.check_response', ['h', 'expected_response', 'tol'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_response', localization, ['h', 'expected_response', 'tol'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_response(...)' code ##################

        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to len(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'h' (line 42)
        h_314073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'h', False)
        # Processing the call keyword arguments (line 42)
        kwargs_314074 = {}
        # Getting the type of 'len' (line 42)
        len_314072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'len', False)
        # Calling len(args, kwargs) (line 42)
        len_call_result_314075 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), len_314072, *[h_314073], **kwargs_314074)
        
        # Assigning a type to the variable 'N' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'N', len_call_result_314075)
        
        # Assigning a BinOp to a Name (line 43):
        
        # Assigning a BinOp to a Name (line 43):
        float_314076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'float')
        # Getting the type of 'N' (line 43)
        N_314077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'N')
        int_314078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'int')
        # Applying the binary operator '-' (line 43)
        result_sub_314079 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '-', N_314077, int_314078)
        
        # Applying the binary operator '*' (line 43)
        result_mul_314080 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 16), '*', float_314076, result_sub_314079)
        
        # Assigning a type to the variable 'alpha' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'alpha', result_mul_314080)
        
        # Assigning a BinOp to a Name (line 44):
        
        # Assigning a BinOp to a Name (line 44):
        
        # Call to arange(...): (line 44)
        # Processing the call arguments (line 44)
        int_314083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'int')
        # Getting the type of 'N' (line 44)
        N_314084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'N', False)
        # Processing the call keyword arguments (line 44)
        kwargs_314085 = {}
        # Getting the type of 'np' (line 44)
        np_314081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 44)
        arange_314082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), np_314081, 'arange')
        # Calling arange(args, kwargs) (line 44)
        arange_call_result_314086 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), arange_314082, *[int_314083, N_314084], **kwargs_314085)
        
        # Getting the type of 'alpha' (line 44)
        alpha_314087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'alpha')
        # Applying the binary operator '-' (line 44)
        result_sub_314088 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '-', arange_call_result_314086, alpha_314087)
        
        # Assigning a type to the variable 'm' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'm', result_sub_314088)
        
        # Getting the type of 'expected_response' (line 45)
        expected_response_314089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'expected_response')
        # Testing the type of a for loop iterable (line 45)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 8), expected_response_314089)
        # Getting the type of the for loop variable (line 45)
        for_loop_var_314090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 8), expected_response_314089)
        # Assigning a type to the variable 'freq' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'freq', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), for_loop_var_314090))
        # Assigning a type to the variable 'expected' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), for_loop_var_314090))
        # SSA begins for a for statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to abs(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to sum(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'h' (line 46)
        h_314094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 32), 'h', False)
        
        # Call to exp(...): (line 46)
        # Processing the call arguments (line 46)
        complex_314097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 41), 'complex')
        # Getting the type of 'np' (line 46)
        np_314098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'np', False)
        # Obtaining the member 'pi' of a type (line 46)
        pi_314099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 46), np_314098, 'pi')
        # Applying the binary operator '*' (line 46)
        result_mul_314100 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 41), '*', complex_314097, pi_314099)
        
        # Getting the type of 'm' (line 46)
        m_314101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 52), 'm', False)
        # Applying the binary operator '*' (line 46)
        result_mul_314102 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 51), '*', result_mul_314100, m_314101)
        
        # Getting the type of 'freq' (line 46)
        freq_314103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 'freq', False)
        # Applying the binary operator '*' (line 46)
        result_mul_314104 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 53), '*', result_mul_314102, freq_314103)
        
        # Processing the call keyword arguments (line 46)
        kwargs_314105 = {}
        # Getting the type of 'np' (line 46)
        np_314095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'np', False)
        # Obtaining the member 'exp' of a type (line 46)
        exp_314096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), np_314095, 'exp')
        # Calling exp(args, kwargs) (line 46)
        exp_call_result_314106 = invoke(stypy.reporting.localization.Localization(__file__, 46, 34), exp_314096, *[result_mul_314104], **kwargs_314105)
        
        # Applying the binary operator '*' (line 46)
        result_mul_314107 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 32), '*', h_314094, exp_call_result_314106)
        
        # Processing the call keyword arguments (line 46)
        kwargs_314108 = {}
        # Getting the type of 'np' (line 46)
        np_314092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'np', False)
        # Obtaining the member 'sum' of a type (line 46)
        sum_314093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), np_314092, 'sum')
        # Calling sum(args, kwargs) (line 46)
        sum_call_result_314109 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), sum_314093, *[result_mul_314107], **kwargs_314108)
        
        # Processing the call keyword arguments (line 46)
        kwargs_314110 = {}
        # Getting the type of 'abs' (line 46)
        abs_314091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'abs', False)
        # Calling abs(args, kwargs) (line 46)
        abs_call_result_314111 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), abs_314091, *[sum_call_result_314109], **kwargs_314110)
        
        # Assigning a type to the variable 'actual' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'actual', abs_call_result_314111)
        
        # Assigning a BinOp to a Name (line 47):
        
        # Assigning a BinOp to a Name (line 47):
        
        # Call to abs(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'actual' (line 47)
        actual_314113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'actual', False)
        # Getting the type of 'expected' (line 47)
        expected_314114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'expected', False)
        # Applying the binary operator '-' (line 47)
        result_sub_314115 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 22), '-', actual_314113, expected_314114)
        
        # Processing the call keyword arguments (line 47)
        kwargs_314116 = {}
        # Getting the type of 'abs' (line 47)
        abs_314112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'abs', False)
        # Calling abs(args, kwargs) (line 47)
        abs_call_result_314117 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), abs_314112, *[result_sub_314115], **kwargs_314116)
        
        int_314118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 40), 'int')
        # Applying the binary operator '**' (line 47)
        result_pow_314119 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 18), '**', abs_call_result_314117, int_314118)
        
        # Assigning a type to the variable 'mse' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'mse', result_pow_314119)
        
        # Call to assert_(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Getting the type of 'mse' (line 48)
        mse_314121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'mse', False)
        # Getting the type of 'tol' (line 48)
        tol_314122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'tol', False)
        # Applying the binary operator '<' (line 48)
        result_lt_314123 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 20), '<', mse_314121, tol_314122)
        
        str_314124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'str', 'response not as expected, mse=%g > %g')
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_314125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'mse' (line 49)
        mse_314126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'mse', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), tuple_314125, mse_314126)
        # Adding element type (line 49)
        # Getting the type of 'tol' (line 49)
        tol_314127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'tol', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 18), tuple_314125, tol_314127)
        
        # Applying the binary operator '%' (line 48)
        result_mod_314128 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 31), '%', str_314124, tuple_314125)
        
        # Processing the call keyword arguments (line 48)
        kwargs_314129 = {}
        # Getting the type of 'assert_' (line 48)
        assert__314120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 48)
        assert__call_result_314130 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), assert__314120, *[result_lt_314123, result_mod_314128], **kwargs_314129)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_response(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_response' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_314131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314131)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_response'
        return stypy_return_type_314131


    @norecursion
    def test_response(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_response'
        module_type_store = module_type_store.open_function_context('test_response', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin.test_response.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin.test_response.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin.test_response.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin.test_response.__dict__.__setitem__('stypy_function_name', 'TestFirwin.test_response')
        TestFirwin.test_response.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin.test_response.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin.test_response.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin.test_response.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin.test_response.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin.test_response.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin.test_response.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin.test_response', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_response', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_response(...)' code ##################

        
        # Assigning a Num to a Name (line 52):
        
        # Assigning a Num to a Name (line 52):
        int_314132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'N' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'N', int_314132)
        
        # Assigning a Num to a Name (line 53):
        
        # Assigning a Num to a Name (line 53):
        float_314133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 12), 'float')
        # Assigning a type to the variable 'f' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'f', float_314133)
        
        # Assigning a Call to a Name (line 55):
        
        # Assigning a Call to a Name (line 55):
        
        # Call to firwin(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'N' (line 55)
        N_314135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'N', False)
        # Getting the type of 'f' (line 55)
        f_314136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'f', False)
        # Processing the call keyword arguments (line 55)
        kwargs_314137 = {}
        # Getting the type of 'firwin' (line 55)
        firwin_314134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 55)
        firwin_call_result_314138 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), firwin_314134, *[N_314135, f_314136], **kwargs_314137)
        
        # Assigning a type to the variable 'h' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'h', firwin_call_result_314138)
        
        # Call to check_response(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'h' (line 56)
        h_314141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_314142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_314143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        float_314144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 33), tuple_314143, float_314144)
        # Adding element type (line 56)
        int_314145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 33), tuple_314143, int_314145)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 31), list_314142, tuple_314143)
        # Adding element type (line 56)
        
        # Obtaining an instance of the builtin type 'tuple' (line 56)
        tuple_314146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 56)
        # Adding element type (line 56)
        float_314147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 42), tuple_314146, float_314147)
        # Adding element type (line 56)
        int_314148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 42), tuple_314146, int_314148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 31), list_314142, tuple_314146)
        
        # Processing the call keyword arguments (line 56)
        kwargs_314149 = {}
        # Getting the type of 'self' (line 56)
        self_314139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 56)
        check_response_314140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_314139, 'check_response')
        # Calling check_response(args, kwargs) (line 56)
        check_response_call_result_314150 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), check_response_314140, *[h_314141, list_314142], **kwargs_314149)
        
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to firwin(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'N' (line 58)
        N_314152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'N', False)
        int_314153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
        # Applying the binary operator '+' (line 58)
        result_add_314154 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), '+', N_314152, int_314153)
        
        # Getting the type of 'f' (line 58)
        f_314155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'f', False)
        # Processing the call keyword arguments (line 58)
        str_314156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 34), 'str', 'nuttall')
        keyword_314157 = str_314156
        kwargs_314158 = {'window': keyword_314157}
        # Getting the type of 'firwin' (line 58)
        firwin_314151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 58)
        firwin_call_result_314159 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), firwin_314151, *[result_add_314154, f_314155], **kwargs_314158)
        
        # Assigning a type to the variable 'h' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'h', firwin_call_result_314159)
        
        # Call to check_response(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'h' (line 59)
        h_314162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_314163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        # Adding element type (line 59)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_314164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        float_314165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 33), tuple_314164, float_314165)
        # Adding element type (line 59)
        int_314166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 33), tuple_314164, int_314166)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 31), list_314163, tuple_314164)
        # Adding element type (line 59)
        
        # Obtaining an instance of the builtin type 'tuple' (line 59)
        tuple_314167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 59)
        # Adding element type (line 59)
        float_314168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 42), tuple_314167, float_314168)
        # Adding element type (line 59)
        int_314169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 42), tuple_314167, int_314169)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 31), list_314163, tuple_314167)
        
        # Processing the call keyword arguments (line 59)
        kwargs_314170 = {}
        # Getting the type of 'self' (line 59)
        self_314160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 59)
        check_response_314161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_314160, 'check_response')
        # Calling check_response(args, kwargs) (line 59)
        check_response_call_result_314171 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), check_response_314161, *[h_314162, list_314163], **kwargs_314170)
        
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to firwin(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'N' (line 61)
        N_314173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'N', False)
        int_314174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 21), 'int')
        # Applying the binary operator '+' (line 61)
        result_add_314175 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 19), '+', N_314173, int_314174)
        
        # Getting the type of 'f' (line 61)
        f_314176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'f', False)
        # Processing the call keyword arguments (line 61)
        # Getting the type of 'False' (line 61)
        False_314177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 37), 'False', False)
        keyword_314178 = False_314177
        kwargs_314179 = {'pass_zero': keyword_314178}
        # Getting the type of 'firwin' (line 61)
        firwin_314172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 61)
        firwin_call_result_314180 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), firwin_314172, *[result_add_314175, f_314176], **kwargs_314179)
        
        # Assigning a type to the variable 'h' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'h', firwin_call_result_314180)
        
        # Call to check_response(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'h' (line 62)
        h_314183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_314184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_314185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        float_314186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 33), tuple_314185, float_314186)
        # Adding element type (line 62)
        int_314187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 33), tuple_314185, int_314187)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 31), list_314184, tuple_314185)
        # Adding element type (line 62)
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_314188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        float_314189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), tuple_314188, float_314189)
        # Adding element type (line 62)
        int_314190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 42), tuple_314188, int_314190)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 31), list_314184, tuple_314188)
        
        # Processing the call keyword arguments (line 62)
        kwargs_314191 = {}
        # Getting the type of 'self' (line 62)
        self_314181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 62)
        check_response_314182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_314181, 'check_response')
        # Calling check_response(args, kwargs) (line 62)
        check_response_call_result_314192 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), check_response_314182, *[h_314183, list_314184], **kwargs_314191)
        
        
        # Assigning a Tuple to a Tuple (line 64):
        
        # Assigning a Num to a Name (line 64):
        float_314193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'float')
        # Assigning a type to the variable 'tuple_assignment_313903' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313903', float_314193)
        
        # Assigning a Num to a Name (line 64):
        float_314194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 29), 'float')
        # Assigning a type to the variable 'tuple_assignment_313904' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313904', float_314194)
        
        # Assigning a Num to a Name (line 64):
        float_314195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'float')
        # Assigning a type to the variable 'tuple_assignment_313905' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313905', float_314195)
        
        # Assigning a Num to a Name (line 64):
        float_314196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'float')
        # Assigning a type to the variable 'tuple_assignment_313906' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313906', float_314196)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_assignment_313903' (line 64)
        tuple_assignment_313903_314197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313903')
        # Assigning a type to the variable 'f1' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'f1', tuple_assignment_313903_314197)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_assignment_313904' (line 64)
        tuple_assignment_313904_314198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313904')
        # Assigning a type to the variable 'f2' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'f2', tuple_assignment_313904_314198)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_assignment_313905' (line 64)
        tuple_assignment_313905_314199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313905')
        # Assigning a type to the variable 'f3' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'f3', tuple_assignment_313905_314199)
        
        # Assigning a Name to a Name (line 64):
        # Getting the type of 'tuple_assignment_313906' (line 64)
        tuple_assignment_313906_314200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'tuple_assignment_313906')
        # Assigning a type to the variable 'f4' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'f4', tuple_assignment_313906_314200)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to firwin(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'N' (line 65)
        N_314202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'N', False)
        int_314203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
        # Applying the binary operator '+' (line 65)
        result_add_314204 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '+', N_314202, int_314203)
        
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_314205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        # Getting the type of 'f1' (line 65)
        f1_314206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'f1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_314205, f1_314206)
        # Adding element type (line 65)
        # Getting the type of 'f2' (line 65)
        f2_314207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'f2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 24), list_314205, f2_314207)
        
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'False' (line 65)
        False_314208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'False', False)
        keyword_314209 = False_314208
        kwargs_314210 = {'pass_zero': keyword_314209}
        # Getting the type of 'firwin' (line 65)
        firwin_314201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 65)
        firwin_call_result_314211 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), firwin_314201, *[result_add_314204, list_314205], **kwargs_314210)
        
        # Assigning a type to the variable 'h' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'h', firwin_call_result_314211)
        
        # Call to check_response(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'h' (line 66)
        h_314214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_314215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_314216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        float_314217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 33), tuple_314216, float_314217)
        # Adding element type (line 66)
        int_314218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 33), tuple_314216, int_314218)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 31), list_314215, tuple_314216)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_314219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        float_314220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 41), tuple_314219, float_314220)
        # Adding element type (line 66)
        int_314221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 41), tuple_314219, int_314221)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 31), list_314215, tuple_314219)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'tuple' (line 66)
        tuple_314222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 66)
        # Adding element type (line 66)
        float_314223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 49), tuple_314222, float_314223)
        # Adding element type (line 66)
        int_314224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 49), tuple_314222, int_314224)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 31), list_314215, tuple_314222)
        
        # Processing the call keyword arguments (line 66)
        kwargs_314225 = {}
        # Getting the type of 'self' (line 66)
        self_314212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 66)
        check_response_314213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_314212, 'check_response')
        # Calling check_response(args, kwargs) (line 66)
        check_response_call_result_314226 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), check_response_314213, *[h_314214, list_314215], **kwargs_314225)
        
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to firwin(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'N' (line 68)
        N_314228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'N', False)
        int_314229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'int')
        # Applying the binary operator '+' (line 68)
        result_add_314230 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 19), '+', N_314228, int_314229)
        
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_314231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'f1' (line 68)
        f1_314232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'f1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 24), list_314231, f1_314232)
        # Adding element type (line 68)
        # Getting the type of 'f2' (line 68)
        f2_314233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'f2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 24), list_314231, f2_314233)
        
        # Processing the call keyword arguments (line 68)
        kwargs_314234 = {}
        # Getting the type of 'firwin' (line 68)
        firwin_314227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 68)
        firwin_call_result_314235 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), firwin_314227, *[result_add_314230, list_314231], **kwargs_314234)
        
        # Assigning a type to the variable 'h' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'h', firwin_call_result_314235)
        
        # Call to check_response(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'h' (line 69)
        h_314238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_314239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_314240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        float_314241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), tuple_314240, float_314241)
        # Adding element type (line 69)
        int_314242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), tuple_314240, int_314242)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_314239, tuple_314240)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_314243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        float_314244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 41), tuple_314243, float_314244)
        # Adding element type (line 69)
        int_314245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 41), tuple_314243, int_314245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_314239, tuple_314243)
        # Adding element type (line 69)
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_314246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        float_314247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 49), tuple_314246, float_314247)
        # Adding element type (line 69)
        int_314248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 49), tuple_314246, int_314248)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_314239, tuple_314246)
        
        # Processing the call keyword arguments (line 69)
        kwargs_314249 = {}
        # Getting the type of 'self' (line 69)
        self_314236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 69)
        check_response_314237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_314236, 'check_response')
        # Calling check_response(args, kwargs) (line 69)
        check_response_call_result_314250 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), check_response_314237, *[h_314238, list_314239], **kwargs_314249)
        
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to firwin(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'N' (line 71)
        N_314252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'N', False)
        int_314253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'int')
        # Applying the binary operator '+' (line 71)
        result_add_314254 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), '+', N_314252, int_314253)
        
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_314255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'f1' (line 71)
        f1_314256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'f1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), list_314255, f1_314256)
        # Adding element type (line 71)
        # Getting the type of 'f2' (line 71)
        f2_314257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'f2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), list_314255, f2_314257)
        # Adding element type (line 71)
        # Getting the type of 'f3' (line 71)
        f3_314258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'f3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), list_314255, f3_314258)
        # Adding element type (line 71)
        # Getting the type of 'f4' (line 71)
        f4_314259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'f4', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 24), list_314255, f4_314259)
        
        # Processing the call keyword arguments (line 71)
        # Getting the type of 'False' (line 71)
        False_314260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 52), 'False', False)
        keyword_314261 = False_314260
        # Getting the type of 'False' (line 71)
        False_314262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 65), 'False', False)
        keyword_314263 = False_314262
        kwargs_314264 = {'pass_zero': keyword_314261, 'scale': keyword_314263}
        # Getting the type of 'firwin' (line 71)
        firwin_314251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 71)
        firwin_call_result_314265 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), firwin_314251, *[result_add_314254, list_314255], **kwargs_314264)
        
        # Assigning a type to the variable 'h' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'h', firwin_call_result_314265)
        
        # Call to check_response(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'h' (line 72)
        h_314268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_314269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_314270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        float_314271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 33), tuple_314270, float_314271)
        # Adding element type (line 72)
        int_314272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 33), tuple_314270, int_314272)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_314269, tuple_314270)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_314273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        float_314274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 41), tuple_314273, float_314274)
        # Adding element type (line 72)
        int_314275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 41), tuple_314273, int_314275)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_314269, tuple_314273)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_314276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        float_314277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 49), tuple_314276, float_314277)
        # Adding element type (line 72)
        int_314278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 49), tuple_314276, int_314278)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_314269, tuple_314276)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_314279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        float_314280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 57), tuple_314279, float_314280)
        # Adding element type (line 72)
        int_314281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 57), tuple_314279, int_314281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_314269, tuple_314279)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_314282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        float_314283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 65), tuple_314282, float_314283)
        # Adding element type (line 72)
        int_314284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 65), tuple_314282, int_314284)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 31), list_314269, tuple_314282)
        
        # Processing the call keyword arguments (line 72)
        kwargs_314285 = {}
        # Getting the type of 'self' (line 72)
        self_314266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 72)
        check_response_314267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_314266, 'check_response')
        # Calling check_response(args, kwargs) (line 72)
        check_response_call_result_314286 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), check_response_314267, *[h_314268, list_314269], **kwargs_314285)
        
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to firwin(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'N' (line 74)
        N_314288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'N', False)
        int_314289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'int')
        # Applying the binary operator '+' (line 74)
        result_add_314290 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '+', N_314288, int_314289)
        
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_314291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        # Getting the type of 'f1' (line 74)
        f1_314292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'f1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_314291, f1_314292)
        # Adding element type (line 74)
        # Getting the type of 'f2' (line 74)
        f2_314293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), 'f2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_314291, f2_314293)
        # Adding element type (line 74)
        # Getting the type of 'f3' (line 74)
        f3_314294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'f3', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_314291, f3_314294)
        # Adding element type (line 74)
        # Getting the type of 'f4' (line 74)
        f4_314295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 37), 'f4', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_314291, f4_314295)
        
        # Processing the call keyword arguments (line 74)
        kwargs_314296 = {}
        # Getting the type of 'firwin' (line 74)
        firwin_314287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 74)
        firwin_call_result_314297 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), firwin_314287, *[result_add_314290, list_314291], **kwargs_314296)
        
        # Assigning a type to the variable 'h' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'h', firwin_call_result_314297)
        
        # Call to check_response(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'h' (line 75)
        h_314300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_314301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_314302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_314303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 33), tuple_314302, float_314303)
        # Adding element type (line 75)
        int_314304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 33), tuple_314302, int_314304)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_314301, tuple_314302)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_314305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_314306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 41), tuple_314305, float_314306)
        # Adding element type (line 75)
        int_314307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 41), tuple_314305, int_314307)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_314301, tuple_314305)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_314308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_314309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 49), tuple_314308, float_314309)
        # Adding element type (line 75)
        int_314310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 49), tuple_314308, int_314310)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_314301, tuple_314308)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_314311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_314312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 57), tuple_314311, float_314312)
        # Adding element type (line 75)
        int_314313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 57), tuple_314311, int_314313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_314301, tuple_314311)
        # Adding element type (line 75)
        
        # Obtaining an instance of the builtin type 'tuple' (line 75)
        tuple_314314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 65), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 75)
        # Adding element type (line 75)
        float_314315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 65), tuple_314314, float_314315)
        # Adding element type (line 75)
        int_314316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 65), tuple_314314, int_314316)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 31), list_314301, tuple_314314)
        
        # Processing the call keyword arguments (line 75)
        kwargs_314317 = {}
        # Getting the type of 'self' (line 75)
        self_314298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 75)
        check_response_314299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_314298, 'check_response')
        # Calling check_response(args, kwargs) (line 75)
        check_response_call_result_314318 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), check_response_314299, *[h_314300, list_314301], **kwargs_314317)
        
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to firwin(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'N' (line 77)
        N_314320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'N', False)
        int_314321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'int')
        # Applying the binary operator '+' (line 77)
        result_add_314322 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 19), '+', N_314320, int_314321)
        
        float_314323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 24), 'float')
        # Processing the call keyword arguments (line 77)
        float_314324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'float')
        keyword_314325 = float_314324
        kwargs_314326 = {'width': keyword_314325}
        # Getting the type of 'firwin' (line 77)
        firwin_314319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 77)
        firwin_call_result_314327 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), firwin_314319, *[result_add_314322, float_314323], **kwargs_314326)
        
        # Assigning a type to the variable 'h' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'h', firwin_call_result_314327)
        
        # Call to check_response(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'h' (line 78)
        h_314330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_314331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_314332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        float_314333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 33), tuple_314332, float_314333)
        # Adding element type (line 78)
        int_314334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 33), tuple_314332, int_314334)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 31), list_314331, tuple_314332)
        # Adding element type (line 78)
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_314335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        float_314336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 42), tuple_314335, float_314336)
        # Adding element type (line 78)
        int_314337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 42), tuple_314335, int_314337)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 31), list_314331, tuple_314335)
        
        # Processing the call keyword arguments (line 78)
        kwargs_314338 = {}
        # Getting the type of 'self' (line 78)
        self_314328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 78)
        check_response_314329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_314328, 'check_response')
        # Calling check_response(args, kwargs) (line 78)
        check_response_call_result_314339 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), check_response_314329, *[h_314330, list_314331], **kwargs_314338)
        
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to firwin(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'N' (line 80)
        N_314341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'N', False)
        int_314342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
        # Applying the binary operator '+' (line 80)
        result_add_314343 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 19), '+', N_314341, int_314342)
        
        float_314344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'float')
        # Processing the call keyword arguments (line 80)
        # Getting the type of 'False' (line 80)
        False_314345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 39), 'False', False)
        keyword_314346 = False_314345
        kwargs_314347 = {'pass_zero': keyword_314346}
        # Getting the type of 'firwin' (line 80)
        firwin_314340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'firwin', False)
        # Calling firwin(args, kwargs) (line 80)
        firwin_call_result_314348 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), firwin_314340, *[result_add_314343, float_314344], **kwargs_314347)
        
        # Assigning a type to the variable 'h' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'h', firwin_call_result_314348)
        
        # Call to check_response(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'h' (line 81)
        h_314351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_314352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_314353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        float_314354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 33), tuple_314353, float_314354)
        # Adding element type (line 81)
        int_314355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 33), tuple_314353, int_314355)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 31), list_314352, tuple_314353)
        # Adding element type (line 81)
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_314356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        float_314357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 42), tuple_314356, float_314357)
        # Adding element type (line 81)
        int_314358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 42), tuple_314356, int_314358)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 31), list_314352, tuple_314356)
        
        # Processing the call keyword arguments (line 81)
        kwargs_314359 = {}
        # Getting the type of 'self' (line 81)
        self_314349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'check_response' of a type (line 81)
        check_response_314350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_314349, 'check_response')
        # Calling check_response(args, kwargs) (line 81)
        check_response_call_result_314360 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), check_response_314350, *[h_314351, list_314352], **kwargs_314359)
        
        
        # ################# End of 'test_response(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_response' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_314361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_response'
        return stypy_return_type_314361


    @norecursion
    def mse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mse'
        module_type_store = module_type_store.open_function_context('mse', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin.mse.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin.mse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin.mse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin.mse.__dict__.__setitem__('stypy_function_name', 'TestFirwin.mse')
        TestFirwin.mse.__dict__.__setitem__('stypy_param_names_list', ['h', 'bands'])
        TestFirwin.mse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin.mse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin.mse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin.mse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin.mse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin.mse.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin.mse', ['h', 'bands'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mse', localization, ['h', 'bands'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mse(...)' code ##################

        str_314362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'Compute mean squared error versus ideal response across frequency\n        band.\n          h -- coefficients\n          bands -- list of (left, right) tuples relative to 1==Nyquist of\n            passbands\n        ')
        
        # Assigning a Call to a Tuple (line 90):
        
        # Assigning a Subscript to a Name (line 90):
        
        # Obtaining the type of the subscript
        int_314363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'int')
        
        # Call to freqz(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'h' (line 90)
        h_314365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'h', False)
        # Processing the call keyword arguments (line 90)
        int_314366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'int')
        keyword_314367 = int_314366
        kwargs_314368 = {'worN': keyword_314367}
        # Getting the type of 'freqz' (line 90)
        freqz_314364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 90)
        freqz_call_result_314369 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), freqz_314364, *[h_314365], **kwargs_314368)
        
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___314370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), freqz_call_result_314369, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_314371 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), getitem___314370, int_314363)
        
        # Assigning a type to the variable 'tuple_var_assignment_313907' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'tuple_var_assignment_313907', subscript_call_result_314371)
        
        # Assigning a Subscript to a Name (line 90):
        
        # Obtaining the type of the subscript
        int_314372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'int')
        
        # Call to freqz(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'h' (line 90)
        h_314374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'h', False)
        # Processing the call keyword arguments (line 90)
        int_314375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'int')
        keyword_314376 = int_314375
        kwargs_314377 = {'worN': keyword_314376}
        # Getting the type of 'freqz' (line 90)
        freqz_314373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 90)
        freqz_call_result_314378 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), freqz_314373, *[h_314374], **kwargs_314377)
        
        # Obtaining the member '__getitem__' of a type (line 90)
        getitem___314379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), freqz_call_result_314378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 90)
        subscript_call_result_314380 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), getitem___314379, int_314372)
        
        # Assigning a type to the variable 'tuple_var_assignment_313908' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'tuple_var_assignment_313908', subscript_call_result_314380)
        
        # Assigning a Name to a Name (line 90):
        # Getting the type of 'tuple_var_assignment_313907' (line 90)
        tuple_var_assignment_313907_314381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'tuple_var_assignment_313907')
        # Assigning a type to the variable 'w' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'w', tuple_var_assignment_313907_314381)
        
        # Assigning a Name to a Name (line 90):
        # Getting the type of 'tuple_var_assignment_313908' (line 90)
        tuple_var_assignment_313908_314382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'tuple_var_assignment_313908')
        # Assigning a type to the variable 'H' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'H', tuple_var_assignment_313908_314382)
        
        # Assigning a BinOp to a Name (line 91):
        
        # Assigning a BinOp to a Name (line 91):
        # Getting the type of 'w' (line 91)
        w_314383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'w')
        # Getting the type of 'np' (line 91)
        np_314384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'np')
        # Obtaining the member 'pi' of a type (line 91)
        pi_314385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 14), np_314384, 'pi')
        # Applying the binary operator 'div' (line 91)
        result_div_314386 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), 'div', w_314383, pi_314385)
        
        # Assigning a type to the variable 'f' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'f', result_div_314386)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to zeros(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to len(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'w' (line 92)
        w_314390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'w', False)
        # Processing the call keyword arguments (line 92)
        kwargs_314391 = {}
        # Getting the type of 'len' (line 92)
        len_314389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'len', False)
        # Calling len(args, kwargs) (line 92)
        len_call_result_314392 = invoke(stypy.reporting.localization.Localization(__file__, 92, 33), len_314389, *[w_314390], **kwargs_314391)
        
        # Getting the type of 'bool' (line 92)
        bool_314393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'bool', False)
        # Processing the call keyword arguments (line 92)
        kwargs_314394 = {}
        # Getting the type of 'np' (line 92)
        np_314387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'np', False)
        # Obtaining the member 'zeros' of a type (line 92)
        zeros_314388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), np_314387, 'zeros')
        # Calling zeros(args, kwargs) (line 92)
        zeros_call_result_314395 = invoke(stypy.reporting.localization.Localization(__file__, 92, 24), zeros_314388, *[len_call_result_314392, bool_314393], **kwargs_314394)
        
        # Assigning a type to the variable 'passIndicator' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'passIndicator', zeros_call_result_314395)
        
        # Getting the type of 'bands' (line 93)
        bands_314396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'bands')
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), bands_314396)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_314397 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), bands_314396)
        # Assigning a type to the variable 'left' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'left', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 8), for_loop_var_314397))
        # Assigning a type to the variable 'right' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'right', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 8), for_loop_var_314397))
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'passIndicator' (line 94)
        passIndicator_314398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'passIndicator')
        
        # Getting the type of 'f' (line 94)
        f_314399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 30), 'f')
        # Getting the type of 'left' (line 94)
        left_314400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'left')
        # Applying the binary operator '>=' (line 94)
        result_ge_314401 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 30), '>=', f_314399, left_314400)
        
        
        # Getting the type of 'f' (line 94)
        f_314402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 44), 'f')
        # Getting the type of 'right' (line 94)
        right_314403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 48), 'right')
        # Applying the binary operator '<' (line 94)
        result_lt_314404 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 44), '<', f_314402, right_314403)
        
        # Applying the binary operator '&' (line 94)
        result_and__314405 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 29), '&', result_ge_314401, result_lt_314404)
        
        # Applying the binary operator '|=' (line 94)
        result_ior_314406 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), '|=', passIndicator_314398, result_and__314405)
        # Assigning a type to the variable 'passIndicator' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'passIndicator', result_ior_314406)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to where(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'passIndicator' (line 95)
        passIndicator_314409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'passIndicator', False)
        int_314410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 41), 'int')
        int_314411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 44), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_314412 = {}
        # Getting the type of 'np' (line 95)
        np_314407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'np', False)
        # Obtaining the member 'where' of a type (line 95)
        where_314408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 17), np_314407, 'where')
        # Calling where(args, kwargs) (line 95)
        where_call_result_314413 = invoke(stypy.reporting.localization.Localization(__file__, 95, 17), where_314408, *[passIndicator_314409, int_314410, int_314411], **kwargs_314412)
        
        # Assigning a type to the variable 'Hideal' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'Hideal', where_call_result_314413)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to mean(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to abs(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to abs(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'H' (line 96)
        H_314418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'H', False)
        # Processing the call keyword arguments (line 96)
        kwargs_314419 = {}
        # Getting the type of 'abs' (line 96)
        abs_314417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'abs', False)
        # Calling abs(args, kwargs) (line 96)
        abs_call_result_314420 = invoke(stypy.reporting.localization.Localization(__file__, 96, 26), abs_314417, *[H_314418], **kwargs_314419)
        
        # Getting the type of 'Hideal' (line 96)
        Hideal_314421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'Hideal', False)
        # Applying the binary operator '-' (line 96)
        result_sub_314422 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 26), '-', abs_call_result_314420, Hideal_314421)
        
        # Processing the call keyword arguments (line 96)
        kwargs_314423 = {}
        # Getting the type of 'abs' (line 96)
        abs_314416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'abs', False)
        # Calling abs(args, kwargs) (line 96)
        abs_call_result_314424 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), abs_314416, *[result_sub_314422], **kwargs_314423)
        
        int_314425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 42), 'int')
        # Applying the binary operator '**' (line 96)
        result_pow_314426 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), '**', abs_call_result_314424, int_314425)
        
        # Processing the call keyword arguments (line 96)
        kwargs_314427 = {}
        # Getting the type of 'np' (line 96)
        np_314414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'np', False)
        # Obtaining the member 'mean' of a type (line 96)
        mean_314415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), np_314414, 'mean')
        # Calling mean(args, kwargs) (line 96)
        mean_call_result_314428 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), mean_314415, *[result_pow_314426], **kwargs_314427)
        
        # Assigning a type to the variable 'mse' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'mse', mean_call_result_314428)
        # Getting the type of 'mse' (line 97)
        mse_314429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'mse')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', mse_314429)
        
        # ################# End of 'mse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mse' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_314430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mse'
        return stypy_return_type_314430


    @norecursion
    def test_scaling(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scaling'
        module_type_store = module_type_store.open_function_context('test_scaling', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_function_name', 'TestFirwin.test_scaling')
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin.test_scaling.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin.test_scaling', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scaling', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scaling(...)' code ##################

        str_314431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n        For one lowpass, bandpass, and highpass example filter, this test\n        checks two things:\n          - the mean squared error over the frequency domain of the unscaled\n            filter is smaller than the scaled filter (true for rectangular\n            window)\n          - the response of the scaled filter is exactly unity at the center\n            of the first passband\n        ')
        
        # Assigning a Num to a Name (line 109):
        
        # Assigning a Num to a Name (line 109):
        int_314432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
        # Assigning a type to the variable 'N' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'N', int_314432)
        
        # Assigning a List to a Name (line 110):
        
        # Assigning a List to a Name (line 110):
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_314433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_314434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_314435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        float_314436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 13), list_314435, float_314436)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 13), tuple_314434, list_314435)
        # Adding element type (line 111)
        # Getting the type of 'True' (line 111)
        True_314437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 13), tuple_314434, True_314437)
        # Adding element type (line 111)
        
        # Obtaining an instance of the builtin type 'tuple' (line 111)
        tuple_314438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 111)
        # Adding element type (line 111)
        int_314439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 26), tuple_314438, int_314439)
        # Adding element type (line 111)
        int_314440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 26), tuple_314438, int_314440)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 13), tuple_314434, tuple_314438)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), list_314433, tuple_314434)
        # Adding element type (line 110)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_314441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_314442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        float_314443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 13), list_314442, float_314443)
        # Adding element type (line 112)
        float_314444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 13), list_314442, float_314444)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 13), tuple_314441, list_314442)
        # Adding element type (line 112)
        # Getting the type of 'False' (line 112)
        False_314445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 13), tuple_314441, False_314445)
        # Adding element type (line 112)
        
        # Obtaining an instance of the builtin type 'tuple' (line 112)
        tuple_314446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 112)
        # Adding element type (line 112)
        float_314447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), tuple_314446, float_314447)
        # Adding element type (line 112)
        int_314448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 32), tuple_314446, int_314448)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 13), tuple_314441, tuple_314446)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), list_314433, tuple_314441)
        # Adding element type (line 110)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_314449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_314450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        float_314451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 14), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), list_314450, float_314451)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), tuple_314449, list_314450)
        # Adding element type (line 113)
        # Getting the type of 'False' (line 113)
        False_314452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), tuple_314449, False_314452)
        # Adding element type (line 113)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_314453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        int_314454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 27), tuple_314453, int_314454)
        # Adding element type (line 113)
        int_314455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 27), tuple_314453, int_314455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 13), tuple_314449, tuple_314453)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 16), list_314433, tuple_314449)
        
        # Assigning a type to the variable 'cases' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'cases', list_314433)
        
        # Getting the type of 'cases' (line 115)
        cases_314456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'cases')
        # Testing the type of a for loop iterable (line 115)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), cases_314456)
        # Getting the type of the for loop variable (line 115)
        for_loop_var_314457 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), cases_314456)
        # Assigning a type to the variable 'cutoff' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'cutoff', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_314457))
        # Assigning a type to the variable 'pass_zero' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'pass_zero', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_314457))
        # Assigning a type to the variable 'expected_response' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'expected_response', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_314457))
        # SSA begins for a for statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 116):
        
        # Assigning a Call to a Name (line 116):
        
        # Call to firwin(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'N' (line 116)
        N_314459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'N', False)
        # Getting the type of 'cutoff' (line 116)
        cutoff_314460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'cutoff', False)
        # Processing the call keyword arguments (line 116)
        # Getting the type of 'False' (line 116)
        False_314461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 40), 'False', False)
        keyword_314462 = False_314461
        # Getting the type of 'pass_zero' (line 116)
        pass_zero_314463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 57), 'pass_zero', False)
        keyword_314464 = pass_zero_314463
        str_314465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 75), 'str', 'ones')
        keyword_314466 = str_314465
        kwargs_314467 = {'pass_zero': keyword_314464, 'window': keyword_314466, 'scale': keyword_314462}
        # Getting the type of 'firwin' (line 116)
        firwin_314458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'firwin', False)
        # Calling firwin(args, kwargs) (line 116)
        firwin_call_result_314468 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), firwin_314458, *[N_314459, cutoff_314460], **kwargs_314467)
        
        # Assigning a type to the variable 'h' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'h', firwin_call_result_314468)
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to firwin(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'N' (line 117)
        N_314470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'N', False)
        # Getting the type of 'cutoff' (line 117)
        cutoff_314471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'cutoff', False)
        # Processing the call keyword arguments (line 117)
        # Getting the type of 'True' (line 117)
        True_314472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 41), 'True', False)
        keyword_314473 = True_314472
        # Getting the type of 'pass_zero' (line 117)
        pass_zero_314474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 57), 'pass_zero', False)
        keyword_314475 = pass_zero_314474
        str_314476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 75), 'str', 'ones')
        keyword_314477 = str_314476
        kwargs_314478 = {'pass_zero': keyword_314475, 'window': keyword_314477, 'scale': keyword_314473}
        # Getting the type of 'firwin' (line 117)
        firwin_314469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'firwin', False)
        # Calling firwin(args, kwargs) (line 117)
        firwin_call_result_314479 = invoke(stypy.reporting.localization.Localization(__file__, 117, 17), firwin_314469, *[N_314470, cutoff_314471], **kwargs_314478)
        
        # Assigning a type to the variable 'hs' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'hs', firwin_call_result_314479)
        
        
        
        # Call to len(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'cutoff' (line 118)
        cutoff_314481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'cutoff', False)
        # Processing the call keyword arguments (line 118)
        kwargs_314482 = {}
        # Getting the type of 'len' (line 118)
        len_314480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'len', False)
        # Calling len(args, kwargs) (line 118)
        len_call_result_314483 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), len_314480, *[cutoff_314481], **kwargs_314482)
        
        int_314484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 30), 'int')
        # Applying the binary operator '==' (line 118)
        result_eq_314485 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '==', len_call_result_314483, int_314484)
        
        # Testing the type of an if condition (line 118)
        if_condition_314486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), result_eq_314485)
        # Assigning a type to the variable 'if_condition_314486' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_314486', if_condition_314486)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pass_zero' (line 119)
        pass_zero_314487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'pass_zero')
        # Testing the type of an if condition (line 119)
        if_condition_314488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 16), pass_zero_314487)
        # Assigning a type to the variable 'if_condition_314488' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'if_condition_314488', if_condition_314488)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 120):
        
        # Assigning a BinOp to a Name (line 120):
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_314489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        int_314490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 29), list_314489, int_314490)
        
        # Getting the type of 'cutoff' (line 120)
        cutoff_314491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'cutoff')
        # Applying the binary operator '+' (line 120)
        result_add_314492 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 29), '+', list_314489, cutoff_314491)
        
        # Assigning a type to the variable 'cutoff' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'cutoff', result_add_314492)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        # Getting the type of 'cutoff' (line 122)
        cutoff_314493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'cutoff')
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_314494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        int_314495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 38), list_314494, int_314495)
        
        # Applying the binary operator '+' (line 122)
        result_add_314496 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 29), '+', cutoff_314493, list_314494)
        
        # Assigning a type to the variable 'cutoff' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'cutoff', result_add_314496)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 123)
        # Processing the call arguments (line 123)
        
        
        # Call to mse(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'h' (line 123)
        h_314500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'h', False)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_314501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'cutoff' (line 123)
        cutoff_314502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 33), 'cutoff', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 32), list_314501, cutoff_314502)
        
        # Processing the call keyword arguments (line 123)
        kwargs_314503 = {}
        # Getting the type of 'self' (line 123)
        self_314498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'self', False)
        # Obtaining the member 'mse' of a type (line 123)
        mse_314499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 20), self_314498, 'mse')
        # Calling mse(args, kwargs) (line 123)
        mse_call_result_314504 = invoke(stypy.reporting.localization.Localization(__file__, 123, 20), mse_314499, *[h_314500, list_314501], **kwargs_314503)
        
        
        # Call to mse(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'hs' (line 123)
        hs_314507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'hs', False)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_314508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'cutoff' (line 123)
        cutoff_314509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 58), 'cutoff', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 57), list_314508, cutoff_314509)
        
        # Processing the call keyword arguments (line 123)
        kwargs_314510 = {}
        # Getting the type of 'self' (line 123)
        self_314505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'self', False)
        # Obtaining the member 'mse' of a type (line 123)
        mse_314506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 44), self_314505, 'mse')
        # Calling mse(args, kwargs) (line 123)
        mse_call_result_314511 = invoke(stypy.reporting.localization.Localization(__file__, 123, 44), mse_314506, *[hs_314507, list_314508], **kwargs_314510)
        
        # Applying the binary operator '<' (line 123)
        result_lt_314512 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 20), '<', mse_call_result_314504, mse_call_result_314511)
        
        str_314513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 16), 'str', 'least squares violation')
        # Processing the call keyword arguments (line 123)
        kwargs_314514 = {}
        # Getting the type of 'assert_' (line 123)
        assert__314497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 123)
        assert__call_result_314515 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assert__314497, *[result_lt_314512, str_314513], **kwargs_314514)
        
        
        # Call to check_response(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'hs' (line 125)
        hs_314518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'hs', False)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_314519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'expected_response' (line 125)
        expected_response_314520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'expected_response', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 36), list_314519, expected_response_314520)
        
        float_314521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 57), 'float')
        # Processing the call keyword arguments (line 125)
        kwargs_314522 = {}
        # Getting the type of 'self' (line 125)
        self_314516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'self', False)
        # Obtaining the member 'check_response' of a type (line 125)
        check_response_314517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), self_314516, 'check_response')
        # Calling check_response(args, kwargs) (line 125)
        check_response_call_result_314523 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), check_response_314517, *[hs_314518, list_314519, float_314521], **kwargs_314522)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_scaling(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scaling' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_314524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314524)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scaling'
        return stypy_return_type_314524


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFirwin' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'TestFirwin', TestFirwin)
# Declaration of the 'TestFirWinMore' class

class TestFirWinMore(object, ):
    str_314525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 4), 'str', 'Different author, different style, different tests...')

    @norecursion
    def test_lowpass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lowpass'
        module_type_store = module_type_store.open_function_context('test_lowpass', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_lowpass')
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_lowpass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_lowpass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lowpass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lowpass(...)' code ##################

        
        # Assigning a Num to a Name (line 132):
        
        # Assigning a Num to a Name (line 132):
        float_314526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 16), 'float')
        # Assigning a type to the variable 'width' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'width', float_314526)
        
        # Assigning a Call to a Tuple (line 133):
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_314527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to kaiserord(...): (line 133)
        # Processing the call arguments (line 133)
        int_314529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'int')
        # Getting the type of 'width' (line 133)
        width_314530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'width', False)
        # Processing the call keyword arguments (line 133)
        kwargs_314531 = {}
        # Getting the type of 'kaiserord' (line 133)
        kaiserord_314528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 133)
        kaiserord_call_result_314532 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), kaiserord_314528, *[int_314529, width_314530], **kwargs_314531)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___314533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), kaiserord_call_result_314532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_314534 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___314533, int_314527)
        
        # Assigning a type to the variable 'tuple_var_assignment_313909' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_313909', subscript_call_result_314534)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_314535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to kaiserord(...): (line 133)
        # Processing the call arguments (line 133)
        int_314537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 32), 'int')
        # Getting the type of 'width' (line 133)
        width_314538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'width', False)
        # Processing the call keyword arguments (line 133)
        kwargs_314539 = {}
        # Getting the type of 'kaiserord' (line 133)
        kaiserord_314536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 133)
        kaiserord_call_result_314540 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), kaiserord_314536, *[int_314537, width_314538], **kwargs_314539)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___314541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), kaiserord_call_result_314540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_314542 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___314541, int_314535)
        
        # Assigning a type to the variable 'tuple_var_assignment_313910' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_313910', subscript_call_result_314542)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_313909' (line 133)
        tuple_var_assignment_313909_314543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_313909')
        # Assigning a type to the variable 'ntaps' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'ntaps', tuple_var_assignment_313909_314543)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_313910' (line 133)
        tuple_var_assignment_313910_314544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_313910')
        # Assigning a type to the variable 'beta' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'beta', tuple_var_assignment_313910_314544)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to firwin(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'ntaps' (line 134)
        ntaps_314546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'ntaps', False)
        # Processing the call keyword arguments (line 134)
        float_314547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'float')
        keyword_314548 = float_314547
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_314549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        str_314550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 49), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 49), tuple_314549, str_314550)
        # Adding element type (line 134)
        # Getting the type of 'beta' (line 134)
        beta_314551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 59), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 49), tuple_314549, beta_314551)
        
        keyword_314552 = tuple_314549
        # Getting the type of 'False' (line 134)
        False_314553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 72), 'False', False)
        keyword_314554 = False_314553
        kwargs_314555 = {'cutoff': keyword_314548, 'window': keyword_314552, 'scale': keyword_314554}
        # Getting the type of 'firwin' (line 134)
        firwin_314545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'firwin', False)
        # Calling firwin(args, kwargs) (line 134)
        firwin_call_result_314556 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), firwin_314545, *[ntaps_314546], **kwargs_314555)
        
        # Assigning a type to the variable 'taps' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'taps', firwin_call_result_314556)
        
        # Call to assert_array_almost_equal(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 137)
        ntaps_314558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'ntaps', False)
        int_314559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 47), 'int')
        # Applying the binary operator '//' (line 137)
        result_floordiv_314560 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 40), '//', ntaps_314558, int_314559)
        
        slice_314561 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 137, 34), None, result_floordiv_314560, None)
        # Getting the type of 'taps' (line 137)
        taps_314562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___314563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), taps_314562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_314564 = invoke(stypy.reporting.localization.Localization(__file__, 137, 34), getitem___314563, slice_314561)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 137)
        ntaps_314565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 56), 'ntaps', False)
        # Getting the type of 'ntaps' (line 137)
        ntaps_314566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 62), 'ntaps', False)
        # Getting the type of 'ntaps' (line 137)
        ntaps_314567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 68), 'ntaps', False)
        int_314568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 75), 'int')
        # Applying the binary operator '//' (line 137)
        result_floordiv_314569 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 68), '//', ntaps_314567, int_314568)
        
        # Applying the binary operator '-' (line 137)
        result_sub_314570 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 62), '-', ntaps_314566, result_floordiv_314569)
        
        int_314571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 77), 'int')
        # Applying the binary operator '-' (line 137)
        result_sub_314572 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 76), '-', result_sub_314570, int_314571)
        
        int_314573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 79), 'int')
        slice_314574 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 137, 51), ntaps_314565, result_sub_314572, int_314573)
        # Getting the type of 'taps' (line 137)
        taps_314575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 51), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___314576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 51), taps_314575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_314577 = invoke(stypy.reporting.localization.Localization(__file__, 137, 51), getitem___314576, slice_314574)
        
        # Processing the call keyword arguments (line 137)
        kwargs_314578 = {}
        # Getting the type of 'assert_array_almost_equal' (line 137)
        assert_array_almost_equal_314557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 137)
        assert_array_almost_equal_call_result_314579 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_array_almost_equal_314557, *[subscript_call_result_314564, subscript_call_result_314577], **kwargs_314578)
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to array(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_314582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        # Adding element type (line 140)
        float_314583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, float_314583)
        # Adding element type (line 140)
        float_314584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, float_314584)
        # Adding element type (line 140)
        float_314585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'float')
        # Getting the type of 'width' (line 140)
        width_314586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 48), 'width', False)
        int_314587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 54), 'int')
        # Applying the binary operator 'div' (line 140)
        result_div_314588 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 48), 'div', width_314586, int_314587)
        
        # Applying the binary operator '-' (line 140)
        result_sub_314589 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 44), '-', float_314585, result_div_314588)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, result_sub_314589)
        # Adding element type (line 140)
        float_314590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 57), 'float')
        # Getting the type of 'width' (line 140)
        width_314591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 61), 'width', False)
        int_314592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 67), 'int')
        # Applying the binary operator 'div' (line 140)
        result_div_314593 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 61), 'div', width_314591, int_314592)
        
        # Applying the binary operator '+' (line 140)
        result_add_314594 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 57), '+', float_314590, result_div_314593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, result_add_314594)
        # Adding element type (line 140)
        float_314595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, float_314595)
        # Adding element type (line 140)
        float_314596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 76), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 32), list_314582, float_314596)
        
        # Processing the call keyword arguments (line 140)
        kwargs_314597 = {}
        # Getting the type of 'np' (line 140)
        np_314580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 140)
        array_314581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), np_314580, 'array')
        # Calling array(args, kwargs) (line 140)
        array_call_result_314598 = invoke(stypy.reporting.localization.Localization(__file__, 140, 23), array_314581, *[list_314582], **kwargs_314597)
        
        # Assigning a type to the variable 'freq_samples' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'freq_samples', array_call_result_314598)
        
        # Assigning a Call to a Tuple (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_314599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to freqz(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'taps' (line 141)
        taps_314601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'taps', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'np' (line 141)
        np_314602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 141)
        pi_314603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 43), np_314602, 'pi')
        # Getting the type of 'freq_samples' (line 141)
        freq_samples_314604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 141)
        result_mul_314605 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 43), '*', pi_314603, freq_samples_314604)
        
        keyword_314606 = result_mul_314605
        kwargs_314607 = {'worN': keyword_314606}
        # Getting the type of 'freqz' (line 141)
        freqz_314600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 141)
        freqz_call_result_314608 = invoke(stypy.reporting.localization.Localization(__file__, 141, 26), freqz_314600, *[taps_314601], **kwargs_314607)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___314609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), freqz_call_result_314608, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_314610 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___314609, int_314599)
        
        # Assigning a type to the variable 'tuple_var_assignment_313911' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_313911', subscript_call_result_314610)
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_314611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to freqz(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'taps' (line 141)
        taps_314613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'taps', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'np' (line 141)
        np_314614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 141)
        pi_314615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 43), np_314614, 'pi')
        # Getting the type of 'freq_samples' (line 141)
        freq_samples_314616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 141)
        result_mul_314617 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 43), '*', pi_314615, freq_samples_314616)
        
        keyword_314618 = result_mul_314617
        kwargs_314619 = {'worN': keyword_314618}
        # Getting the type of 'freqz' (line 141)
        freqz_314612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 141)
        freqz_call_result_314620 = invoke(stypy.reporting.localization.Localization(__file__, 141, 26), freqz_314612, *[taps_314613], **kwargs_314619)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___314621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), freqz_call_result_314620, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_314622 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___314621, int_314611)
        
        # Assigning a type to the variable 'tuple_var_assignment_313912' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_313912', subscript_call_result_314622)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_313911' (line 141)
        tuple_var_assignment_313911_314623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_313911')
        # Assigning a type to the variable 'freqs' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'freqs', tuple_var_assignment_313911_314623)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_313912' (line 141)
        tuple_var_assignment_313912_314624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_313912')
        # Assigning a type to the variable 'response' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'response', tuple_var_assignment_313912_314624)
        
        # Call to assert_array_almost_equal(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to abs(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'response' (line 142)
        response_314628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'response', False)
        # Processing the call keyword arguments (line 142)
        kwargs_314629 = {}
        # Getting the type of 'np' (line 142)
        np_314626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 142)
        abs_314627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 34), np_314626, 'abs')
        # Calling abs(args, kwargs) (line 142)
        abs_call_result_314630 = invoke(stypy.reporting.localization.Localization(__file__, 142, 34), abs_314627, *[response_314628], **kwargs_314629)
        
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_314631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        float_314632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314632)
        # Adding element type (line 143)
        float_314633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314633)
        # Adding element type (line 143)
        float_314634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314634)
        # Adding element type (line 143)
        float_314635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314635)
        # Adding element type (line 143)
        float_314636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314636)
        # Adding element type (line 143)
        float_314637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 36), list_314631, float_314637)
        
        # Processing the call keyword arguments (line 142)
        int_314638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 76), 'int')
        keyword_314639 = int_314638
        kwargs_314640 = {'decimal': keyword_314639}
        # Getting the type of 'assert_array_almost_equal' (line 142)
        assert_array_almost_equal_314625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 142)
        assert_array_almost_equal_call_result_314641 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_array_almost_equal_314625, *[abs_call_result_314630, list_314631], **kwargs_314640)
        
        
        # ################# End of 'test_lowpass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lowpass' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_314642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314642)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lowpass'
        return stypy_return_type_314642


    @norecursion
    def test_highpass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_highpass'
        module_type_store = module_type_store.open_function_context('test_highpass', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_highpass')
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_highpass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_highpass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_highpass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_highpass(...)' code ##################

        
        # Assigning a Num to a Name (line 146):
        
        # Assigning a Num to a Name (line 146):
        float_314643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'float')
        # Assigning a type to the variable 'width' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'width', float_314643)
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Subscript to a Name (line 147):
        
        # Obtaining the type of the subscript
        int_314644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        
        # Call to kaiserord(...): (line 147)
        # Processing the call arguments (line 147)
        int_314646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'int')
        # Getting the type of 'width' (line 147)
        width_314647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'width', False)
        # Processing the call keyword arguments (line 147)
        kwargs_314648 = {}
        # Getting the type of 'kaiserord' (line 147)
        kaiserord_314645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 147)
        kaiserord_call_result_314649 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), kaiserord_314645, *[int_314646, width_314647], **kwargs_314648)
        
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___314650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), kaiserord_call_result_314649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_314651 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), getitem___314650, int_314644)
        
        # Assigning a type to the variable 'tuple_var_assignment_313913' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'tuple_var_assignment_313913', subscript_call_result_314651)
        
        # Assigning a Subscript to a Name (line 147):
        
        # Obtaining the type of the subscript
        int_314652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        
        # Call to kaiserord(...): (line 147)
        # Processing the call arguments (line 147)
        int_314654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'int')
        # Getting the type of 'width' (line 147)
        width_314655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'width', False)
        # Processing the call keyword arguments (line 147)
        kwargs_314656 = {}
        # Getting the type of 'kaiserord' (line 147)
        kaiserord_314653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 147)
        kaiserord_call_result_314657 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), kaiserord_314653, *[int_314654, width_314655], **kwargs_314656)
        
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___314658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), kaiserord_call_result_314657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 147)
        subscript_call_result_314659 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), getitem___314658, int_314652)
        
        # Assigning a type to the variable 'tuple_var_assignment_313914' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'tuple_var_assignment_313914', subscript_call_result_314659)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'tuple_var_assignment_313913' (line 147)
        tuple_var_assignment_313913_314660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'tuple_var_assignment_313913')
        # Assigning a type to the variable 'ntaps' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'ntaps', tuple_var_assignment_313913_314660)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'tuple_var_assignment_313914' (line 147)
        tuple_var_assignment_313914_314661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'tuple_var_assignment_313914')
        # Assigning a type to the variable 'beta' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'beta', tuple_var_assignment_313914_314661)
        
        # Getting the type of 'ntaps' (line 150)
        ntaps_314662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'ntaps')
        int_314663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'int')
        # Applying the binary operator '|=' (line 150)
        result_ior_314664 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 8), '|=', ntaps_314662, int_314663)
        # Assigning a type to the variable 'ntaps' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'ntaps', result_ior_314664)
        
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to firwin(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'ntaps' (line 152)
        ntaps_314666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'ntaps', False)
        # Processing the call keyword arguments (line 152)
        float_314667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'float')
        keyword_314668 = float_314667
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_314669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        str_314670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 49), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 49), tuple_314669, str_314670)
        # Adding element type (line 152)
        # Getting the type of 'beta' (line 152)
        beta_314671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 49), tuple_314669, beta_314671)
        
        keyword_314672 = tuple_314669
        # Getting the type of 'False' (line 153)
        False_314673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'False', False)
        keyword_314674 = False_314673
        # Getting the type of 'False' (line 153)
        False_314675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 47), 'False', False)
        keyword_314676 = False_314675
        kwargs_314677 = {'cutoff': keyword_314668, 'window': keyword_314672, 'scale': keyword_314676, 'pass_zero': keyword_314674}
        # Getting the type of 'firwin' (line 152)
        firwin_314665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'firwin', False)
        # Calling firwin(args, kwargs) (line 152)
        firwin_call_result_314678 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), firwin_314665, *[ntaps_314666], **kwargs_314677)
        
        # Assigning a type to the variable 'taps' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'taps', firwin_call_result_314678)
        
        # Call to assert_array_almost_equal(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 156)
        ntaps_314680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 40), 'ntaps', False)
        int_314681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 47), 'int')
        # Applying the binary operator '//' (line 156)
        result_floordiv_314682 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 40), '//', ntaps_314680, int_314681)
        
        slice_314683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 34), None, result_floordiv_314682, None)
        # Getting the type of 'taps' (line 156)
        taps_314684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___314685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 34), taps_314684, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_314686 = invoke(stypy.reporting.localization.Localization(__file__, 156, 34), getitem___314685, slice_314683)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 156)
        ntaps_314687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 56), 'ntaps', False)
        # Getting the type of 'ntaps' (line 156)
        ntaps_314688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 62), 'ntaps', False)
        # Getting the type of 'ntaps' (line 156)
        ntaps_314689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 68), 'ntaps', False)
        int_314690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 75), 'int')
        # Applying the binary operator '//' (line 156)
        result_floordiv_314691 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 68), '//', ntaps_314689, int_314690)
        
        # Applying the binary operator '-' (line 156)
        result_sub_314692 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 62), '-', ntaps_314688, result_floordiv_314691)
        
        int_314693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 77), 'int')
        # Applying the binary operator '-' (line 156)
        result_sub_314694 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 76), '-', result_sub_314692, int_314693)
        
        int_314695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 79), 'int')
        slice_314696 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 51), ntaps_314687, result_sub_314694, int_314695)
        # Getting the type of 'taps' (line 156)
        taps_314697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 51), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___314698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 51), taps_314697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_314699 = invoke(stypy.reporting.localization.Localization(__file__, 156, 51), getitem___314698, slice_314696)
        
        # Processing the call keyword arguments (line 156)
        kwargs_314700 = {}
        # Getting the type of 'assert_array_almost_equal' (line 156)
        assert_array_almost_equal_314679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 156)
        assert_array_almost_equal_call_result_314701 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), assert_array_almost_equal_314679, *[subscript_call_result_314686, subscript_call_result_314699], **kwargs_314700)
        
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to array(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_314704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        float_314705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, float_314705)
        # Adding element type (line 159)
        float_314706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, float_314706)
        # Adding element type (line 159)
        float_314707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 44), 'float')
        # Getting the type of 'width' (line 159)
        width_314708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'width', False)
        int_314709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 54), 'int')
        # Applying the binary operator 'div' (line 159)
        result_div_314710 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 48), 'div', width_314708, int_314709)
        
        # Applying the binary operator '-' (line 159)
        result_sub_314711 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 44), '-', float_314707, result_div_314710)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, result_sub_314711)
        # Adding element type (line 159)
        float_314712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 57), 'float')
        # Getting the type of 'width' (line 159)
        width_314713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 61), 'width', False)
        int_314714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 67), 'int')
        # Applying the binary operator 'div' (line 159)
        result_div_314715 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 61), 'div', width_314713, int_314714)
        
        # Applying the binary operator '+' (line 159)
        result_add_314716 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 57), '+', float_314712, result_div_314715)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, result_add_314716)
        # Adding element type (line 159)
        float_314717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, float_314717)
        # Adding element type (line 159)
        float_314718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 76), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 32), list_314704, float_314718)
        
        # Processing the call keyword arguments (line 159)
        kwargs_314719 = {}
        # Getting the type of 'np' (line 159)
        np_314702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 159)
        array_314703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), np_314702, 'array')
        # Calling array(args, kwargs) (line 159)
        array_call_result_314720 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), array_314703, *[list_314704], **kwargs_314719)
        
        # Assigning a type to the variable 'freq_samples' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'freq_samples', array_call_result_314720)
        
        # Assigning a Call to a Tuple (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_314721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to freqz(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'taps' (line 160)
        taps_314723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'taps', False)
        # Processing the call keyword arguments (line 160)
        # Getting the type of 'np' (line 160)
        np_314724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 160)
        pi_314725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 43), np_314724, 'pi')
        # Getting the type of 'freq_samples' (line 160)
        freq_samples_314726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 160)
        result_mul_314727 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 43), '*', pi_314725, freq_samples_314726)
        
        keyword_314728 = result_mul_314727
        kwargs_314729 = {'worN': keyword_314728}
        # Getting the type of 'freqz' (line 160)
        freqz_314722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 160)
        freqz_call_result_314730 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), freqz_314722, *[taps_314723], **kwargs_314729)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___314731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), freqz_call_result_314730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_314732 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___314731, int_314721)
        
        # Assigning a type to the variable 'tuple_var_assignment_313915' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_313915', subscript_call_result_314732)
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_314733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to freqz(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'taps' (line 160)
        taps_314735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'taps', False)
        # Processing the call keyword arguments (line 160)
        # Getting the type of 'np' (line 160)
        np_314736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 160)
        pi_314737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 43), np_314736, 'pi')
        # Getting the type of 'freq_samples' (line 160)
        freq_samples_314738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 160)
        result_mul_314739 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 43), '*', pi_314737, freq_samples_314738)
        
        keyword_314740 = result_mul_314739
        kwargs_314741 = {'worN': keyword_314740}
        # Getting the type of 'freqz' (line 160)
        freqz_314734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 160)
        freqz_call_result_314742 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), freqz_314734, *[taps_314735], **kwargs_314741)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___314743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), freqz_call_result_314742, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_314744 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___314743, int_314733)
        
        # Assigning a type to the variable 'tuple_var_assignment_313916' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_313916', subscript_call_result_314744)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_313915' (line 160)
        tuple_var_assignment_313915_314745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_313915')
        # Assigning a type to the variable 'freqs' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'freqs', tuple_var_assignment_313915_314745)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_313916' (line 160)
        tuple_var_assignment_313916_314746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_313916')
        # Assigning a type to the variable 'response' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'response', tuple_var_assignment_313916_314746)
        
        # Call to assert_array_almost_equal(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Call to abs(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'response' (line 161)
        response_314750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 41), 'response', False)
        # Processing the call keyword arguments (line 161)
        kwargs_314751 = {}
        # Getting the type of 'np' (line 161)
        np_314748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 161)
        abs_314749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 34), np_314748, 'abs')
        # Calling abs(args, kwargs) (line 161)
        abs_call_result_314752 = invoke(stypy.reporting.localization.Localization(__file__, 161, 34), abs_314749, *[response_314750], **kwargs_314751)
        
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_314753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        float_314754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314754)
        # Adding element type (line 162)
        float_314755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314755)
        # Adding element type (line 162)
        float_314756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314756)
        # Adding element type (line 162)
        float_314757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314757)
        # Adding element type (line 162)
        float_314758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314758)
        # Adding element type (line 162)
        float_314759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), list_314753, float_314759)
        
        # Processing the call keyword arguments (line 161)
        int_314760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 76), 'int')
        keyword_314761 = int_314760
        kwargs_314762 = {'decimal': keyword_314761}
        # Getting the type of 'assert_array_almost_equal' (line 161)
        assert_array_almost_equal_314747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 161)
        assert_array_almost_equal_call_result_314763 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_array_almost_equal_314747, *[abs_call_result_314752, list_314753], **kwargs_314762)
        
        
        # ################# End of 'test_highpass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_highpass' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_314764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_highpass'
        return stypy_return_type_314764


    @norecursion
    def test_bandpass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bandpass'
        module_type_store = module_type_store.open_function_context('test_bandpass', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_bandpass')
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_bandpass.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_bandpass', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bandpass', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bandpass(...)' code ##################

        
        # Assigning a Num to a Name (line 165):
        
        # Assigning a Num to a Name (line 165):
        float_314765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 16), 'float')
        # Assigning a type to the variable 'width' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'width', float_314765)
        
        # Assigning a Call to a Tuple (line 166):
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_314766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
        
        # Call to kaiserord(...): (line 166)
        # Processing the call arguments (line 166)
        int_314768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 32), 'int')
        # Getting the type of 'width' (line 166)
        width_314769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'width', False)
        # Processing the call keyword arguments (line 166)
        kwargs_314770 = {}
        # Getting the type of 'kaiserord' (line 166)
        kaiserord_314767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 166)
        kaiserord_call_result_314771 = invoke(stypy.reporting.localization.Localization(__file__, 166, 22), kaiserord_314767, *[int_314768, width_314769], **kwargs_314770)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___314772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), kaiserord_call_result_314771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_314773 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___314772, int_314766)
        
        # Assigning a type to the variable 'tuple_var_assignment_313917' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_313917', subscript_call_result_314773)
        
        # Assigning a Subscript to a Name (line 166):
        
        # Obtaining the type of the subscript
        int_314774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 8), 'int')
        
        # Call to kaiserord(...): (line 166)
        # Processing the call arguments (line 166)
        int_314776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 32), 'int')
        # Getting the type of 'width' (line 166)
        width_314777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'width', False)
        # Processing the call keyword arguments (line 166)
        kwargs_314778 = {}
        # Getting the type of 'kaiserord' (line 166)
        kaiserord_314775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 166)
        kaiserord_call_result_314779 = invoke(stypy.reporting.localization.Localization(__file__, 166, 22), kaiserord_314775, *[int_314776, width_314777], **kwargs_314778)
        
        # Obtaining the member '__getitem__' of a type (line 166)
        getitem___314780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), kaiserord_call_result_314779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 166)
        subscript_call_result_314781 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), getitem___314780, int_314774)
        
        # Assigning a type to the variable 'tuple_var_assignment_313918' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_313918', subscript_call_result_314781)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_313917' (line 166)
        tuple_var_assignment_313917_314782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_313917')
        # Assigning a type to the variable 'ntaps' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'ntaps', tuple_var_assignment_313917_314782)
        
        # Assigning a Name to a Name (line 166):
        # Getting the type of 'tuple_var_assignment_313918' (line 166)
        tuple_var_assignment_313918_314783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'tuple_var_assignment_313918')
        # Assigning a type to the variable 'beta' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'beta', tuple_var_assignment_313918_314783)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to firwin(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'ntaps' (line 167)
        ntaps_314785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'ntaps', False)
        # Processing the call keyword arguments (line 167)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_314786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        float_314787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 36), list_314786, float_314787)
        # Adding element type (line 167)
        float_314788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 36), list_314786, float_314788)
        
        keyword_314789 = list_314786
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_314790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        str_314791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 56), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 56), tuple_314790, str_314791)
        # Adding element type (line 167)
        # Getting the type of 'beta' (line 167)
        beta_314792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 66), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 56), tuple_314790, beta_314792)
        
        keyword_314793 = tuple_314790
        # Getting the type of 'False' (line 168)
        False_314794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'False', False)
        keyword_314795 = False_314794
        # Getting the type of 'False' (line 168)
        False_314796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 47), 'False', False)
        keyword_314797 = False_314796
        kwargs_314798 = {'cutoff': keyword_314789, 'window': keyword_314793, 'scale': keyword_314797, 'pass_zero': keyword_314795}
        # Getting the type of 'firwin' (line 167)
        firwin_314784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'firwin', False)
        # Calling firwin(args, kwargs) (line 167)
        firwin_call_result_314799 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), firwin_314784, *[ntaps_314785], **kwargs_314798)
        
        # Assigning a type to the variable 'taps' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'taps', firwin_call_result_314799)
        
        # Call to assert_array_almost_equal(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 171)
        ntaps_314801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 40), 'ntaps', False)
        int_314802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 47), 'int')
        # Applying the binary operator '//' (line 171)
        result_floordiv_314803 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 40), '//', ntaps_314801, int_314802)
        
        slice_314804 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 34), None, result_floordiv_314803, None)
        # Getting the type of 'taps' (line 171)
        taps_314805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___314806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 34), taps_314805, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_314807 = invoke(stypy.reporting.localization.Localization(__file__, 171, 34), getitem___314806, slice_314804)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 171)
        ntaps_314808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 56), 'ntaps', False)
        # Getting the type of 'ntaps' (line 171)
        ntaps_314809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 62), 'ntaps', False)
        # Getting the type of 'ntaps' (line 171)
        ntaps_314810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 68), 'ntaps', False)
        int_314811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 75), 'int')
        # Applying the binary operator '//' (line 171)
        result_floordiv_314812 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 68), '//', ntaps_314810, int_314811)
        
        # Applying the binary operator '-' (line 171)
        result_sub_314813 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 62), '-', ntaps_314809, result_floordiv_314812)
        
        int_314814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 77), 'int')
        # Applying the binary operator '-' (line 171)
        result_sub_314815 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 76), '-', result_sub_314813, int_314814)
        
        int_314816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 79), 'int')
        slice_314817 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 171, 51), ntaps_314808, result_sub_314815, int_314816)
        # Getting the type of 'taps' (line 171)
        taps_314818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 51), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___314819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 51), taps_314818, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_314820 = invoke(stypy.reporting.localization.Localization(__file__, 171, 51), getitem___314819, slice_314817)
        
        # Processing the call keyword arguments (line 171)
        kwargs_314821 = {}
        # Getting the type of 'assert_array_almost_equal' (line 171)
        assert_array_almost_equal_314800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 171)
        assert_array_almost_equal_call_result_314822 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_array_almost_equal_314800, *[subscript_call_result_314807, subscript_call_result_314820], **kwargs_314821)
        
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to array(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining an instance of the builtin type 'list' (line 174)
        list_314825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 174)
        # Adding element type (line 174)
        float_314826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, float_314826)
        # Adding element type (line 174)
        float_314827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, float_314827)
        # Adding element type (line 174)
        float_314828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 43), 'float')
        # Getting the type of 'width' (line 174)
        width_314829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 47), 'width', False)
        int_314830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 53), 'int')
        # Applying the binary operator 'div' (line 174)
        result_div_314831 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 47), 'div', width_314829, int_314830)
        
        # Applying the binary operator '-' (line 174)
        result_sub_314832 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 43), '-', float_314828, result_div_314831)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, result_sub_314832)
        # Adding element type (line 174)
        float_314833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 56), 'float')
        # Getting the type of 'width' (line 174)
        width_314834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 60), 'width', False)
        int_314835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 66), 'int')
        # Applying the binary operator 'div' (line 174)
        result_div_314836 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 60), 'div', width_314834, int_314835)
        
        # Applying the binary operator '+' (line 174)
        result_add_314837 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 56), '+', float_314833, result_div_314836)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, result_add_314837)
        # Adding element type (line 174)
        float_314838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, float_314838)
        # Adding element type (line 174)
        float_314839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'float')
        # Getting the type of 'width' (line 175)
        width_314840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'width', False)
        int_314841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 42), 'int')
        # Applying the binary operator 'div' (line 175)
        result_div_314842 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 36), 'div', width_314840, int_314841)
        
        # Applying the binary operator '-' (line 175)
        result_sub_314843 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 32), '-', float_314839, result_div_314842)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, result_sub_314843)
        # Adding element type (line 174)
        float_314844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 45), 'float')
        # Getting the type of 'width' (line 175)
        width_314845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 49), 'width', False)
        int_314846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 55), 'int')
        # Applying the binary operator 'div' (line 175)
        result_div_314847 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 49), 'div', width_314845, int_314846)
        
        # Applying the binary operator '+' (line 175)
        result_add_314848 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 45), '+', float_314844, result_div_314847)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, result_add_314848)
        # Adding element type (line 174)
        float_314849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, float_314849)
        # Adding element type (line 174)
        float_314850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 32), list_314825, float_314850)
        
        # Processing the call keyword arguments (line 174)
        kwargs_314851 = {}
        # Getting the type of 'np' (line 174)
        np_314823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 174)
        array_314824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 23), np_314823, 'array')
        # Calling array(args, kwargs) (line 174)
        array_call_result_314852 = invoke(stypy.reporting.localization.Localization(__file__, 174, 23), array_314824, *[list_314825], **kwargs_314851)
        
        # Assigning a type to the variable 'freq_samples' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'freq_samples', array_call_result_314852)
        
        # Assigning a Call to a Tuple (line 176):
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_314853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to freqz(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'taps' (line 176)
        taps_314855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'taps', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'np' (line 176)
        np_314856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 176)
        pi_314857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 43), np_314856, 'pi')
        # Getting the type of 'freq_samples' (line 176)
        freq_samples_314858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 176)
        result_mul_314859 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 43), '*', pi_314857, freq_samples_314858)
        
        keyword_314860 = result_mul_314859
        kwargs_314861 = {'worN': keyword_314860}
        # Getting the type of 'freqz' (line 176)
        freqz_314854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 176)
        freqz_call_result_314862 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), freqz_314854, *[taps_314855], **kwargs_314861)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___314863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), freqz_call_result_314862, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_314864 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___314863, int_314853)
        
        # Assigning a type to the variable 'tuple_var_assignment_313919' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_313919', subscript_call_result_314864)
        
        # Assigning a Subscript to a Name (line 176):
        
        # Obtaining the type of the subscript
        int_314865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 8), 'int')
        
        # Call to freqz(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'taps' (line 176)
        taps_314867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'taps', False)
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'np' (line 176)
        np_314868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 176)
        pi_314869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 43), np_314868, 'pi')
        # Getting the type of 'freq_samples' (line 176)
        freq_samples_314870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 176)
        result_mul_314871 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 43), '*', pi_314869, freq_samples_314870)
        
        keyword_314872 = result_mul_314871
        kwargs_314873 = {'worN': keyword_314872}
        # Getting the type of 'freqz' (line 176)
        freqz_314866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 176)
        freqz_call_result_314874 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), freqz_314866, *[taps_314867], **kwargs_314873)
        
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___314875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), freqz_call_result_314874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_314876 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), getitem___314875, int_314865)
        
        # Assigning a type to the variable 'tuple_var_assignment_313920' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_313920', subscript_call_result_314876)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_313919' (line 176)
        tuple_var_assignment_313919_314877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_313919')
        # Assigning a type to the variable 'freqs' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'freqs', tuple_var_assignment_313919_314877)
        
        # Assigning a Name to a Name (line 176):
        # Getting the type of 'tuple_var_assignment_313920' (line 176)
        tuple_var_assignment_313920_314878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'tuple_var_assignment_313920')
        # Assigning a type to the variable 'response' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'response', tuple_var_assignment_313920_314878)
        
        # Call to assert_array_almost_equal(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to abs(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'response' (line 177)
        response_314882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 41), 'response', False)
        # Processing the call keyword arguments (line 177)
        kwargs_314883 = {}
        # Getting the type of 'np' (line 177)
        np_314880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 177)
        abs_314881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 34), np_314880, 'abs')
        # Calling abs(args, kwargs) (line 177)
        abs_call_result_314884 = invoke(stypy.reporting.localization.Localization(__file__, 177, 34), abs_314881, *[response_314882], **kwargs_314883)
        
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_314885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        float_314886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314886)
        # Adding element type (line 178)
        float_314887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314887)
        # Adding element type (line 178)
        float_314888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314888)
        # Adding element type (line 178)
        float_314889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314889)
        # Adding element type (line 178)
        float_314890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314890)
        # Adding element type (line 178)
        float_314891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314891)
        # Adding element type (line 178)
        float_314892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314892)
        # Adding element type (line 178)
        float_314893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314893)
        # Adding element type (line 178)
        float_314894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_314885, float_314894)
        
        # Processing the call keyword arguments (line 177)
        int_314895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 71), 'int')
        keyword_314896 = int_314895
        kwargs_314897 = {'decimal': keyword_314896}
        # Getting the type of 'assert_array_almost_equal' (line 177)
        assert_array_almost_equal_314879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 177)
        assert_array_almost_equal_call_result_314898 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assert_array_almost_equal_314879, *[abs_call_result_314884, list_314885], **kwargs_314897)
        
        
        # ################# End of 'test_bandpass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bandpass' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_314899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_314899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bandpass'
        return stypy_return_type_314899


    @norecursion
    def test_multi(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multi'
        module_type_store = module_type_store.open_function_context('test_multi', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_multi')
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_multi.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_multi', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multi', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multi(...)' code ##################

        
        # Assigning a Num to a Name (line 181):
        
        # Assigning a Num to a Name (line 181):
        float_314900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 16), 'float')
        # Assigning a type to the variable 'width' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'width', float_314900)
        
        # Assigning a Call to a Tuple (line 182):
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_314901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to kaiserord(...): (line 182)
        # Processing the call arguments (line 182)
        int_314903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'int')
        # Getting the type of 'width' (line 182)
        width_314904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 37), 'width', False)
        # Processing the call keyword arguments (line 182)
        kwargs_314905 = {}
        # Getting the type of 'kaiserord' (line 182)
        kaiserord_314902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 182)
        kaiserord_call_result_314906 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), kaiserord_314902, *[int_314903, width_314904], **kwargs_314905)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___314907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), kaiserord_call_result_314906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_314908 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___314907, int_314901)
        
        # Assigning a type to the variable 'tuple_var_assignment_313921' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_313921', subscript_call_result_314908)
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_314909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to kaiserord(...): (line 182)
        # Processing the call arguments (line 182)
        int_314911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 32), 'int')
        # Getting the type of 'width' (line 182)
        width_314912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 37), 'width', False)
        # Processing the call keyword arguments (line 182)
        kwargs_314913 = {}
        # Getting the type of 'kaiserord' (line 182)
        kaiserord_314910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 182)
        kaiserord_call_result_314914 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), kaiserord_314910, *[int_314911, width_314912], **kwargs_314913)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___314915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), kaiserord_call_result_314914, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_314916 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___314915, int_314909)
        
        # Assigning a type to the variable 'tuple_var_assignment_313922' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_313922', subscript_call_result_314916)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_313921' (line 182)
        tuple_var_assignment_313921_314917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_313921')
        # Assigning a type to the variable 'ntaps' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'ntaps', tuple_var_assignment_313921_314917)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_313922' (line 182)
        tuple_var_assignment_313922_314918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_313922')
        # Assigning a type to the variable 'beta' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'beta', tuple_var_assignment_313922_314918)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to firwin(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'ntaps' (line 183)
        ntaps_314920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'ntaps', False)
        # Processing the call keyword arguments (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_314921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        float_314922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 36), list_314921, float_314922)
        # Adding element type (line 183)
        float_314923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 36), list_314921, float_314923)
        # Adding element type (line 183)
        float_314924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 36), list_314921, float_314924)
        
        keyword_314925 = list_314921
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_314926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        str_314927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 61), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 61), tuple_314926, str_314927)
        # Adding element type (line 183)
        # Getting the type of 'beta' (line 183)
        beta_314928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 71), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 61), tuple_314926, beta_314928)
        
        keyword_314929 = tuple_314926
        # Getting the type of 'True' (line 184)
        True_314930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'True', False)
        keyword_314931 = True_314930
        # Getting the type of 'False' (line 184)
        False_314932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 46), 'False', False)
        keyword_314933 = False_314932
        kwargs_314934 = {'cutoff': keyword_314925, 'window': keyword_314929, 'scale': keyword_314933, 'pass_zero': keyword_314931}
        # Getting the type of 'firwin' (line 183)
        firwin_314919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'firwin', False)
        # Calling firwin(args, kwargs) (line 183)
        firwin_call_result_314935 = invoke(stypy.reporting.localization.Localization(__file__, 183, 15), firwin_314919, *[ntaps_314920], **kwargs_314934)
        
        # Assigning a type to the variable 'taps' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'taps', firwin_call_result_314935)
        
        # Call to assert_array_almost_equal(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 187)
        ntaps_314937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 40), 'ntaps', False)
        int_314938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 47), 'int')
        # Applying the binary operator '//' (line 187)
        result_floordiv_314939 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 40), '//', ntaps_314937, int_314938)
        
        slice_314940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 34), None, result_floordiv_314939, None)
        # Getting the type of 'taps' (line 187)
        taps_314941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___314942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), taps_314941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_314943 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), getitem___314942, slice_314940)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 187)
        ntaps_314944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 56), 'ntaps', False)
        # Getting the type of 'ntaps' (line 187)
        ntaps_314945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 62), 'ntaps', False)
        # Getting the type of 'ntaps' (line 187)
        ntaps_314946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 68), 'ntaps', False)
        int_314947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 75), 'int')
        # Applying the binary operator '//' (line 187)
        result_floordiv_314948 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 68), '//', ntaps_314946, int_314947)
        
        # Applying the binary operator '-' (line 187)
        result_sub_314949 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 62), '-', ntaps_314945, result_floordiv_314948)
        
        int_314950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 77), 'int')
        # Applying the binary operator '-' (line 187)
        result_sub_314951 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 76), '-', result_sub_314949, int_314950)
        
        int_314952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 79), 'int')
        slice_314953 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 51), ntaps_314944, result_sub_314951, int_314952)
        # Getting the type of 'taps' (line 187)
        taps_314954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 51), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___314955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 51), taps_314954, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_314956 = invoke(stypy.reporting.localization.Localization(__file__, 187, 51), getitem___314955, slice_314953)
        
        # Processing the call keyword arguments (line 187)
        kwargs_314957 = {}
        # Getting the type of 'assert_array_almost_equal' (line 187)
        assert_array_almost_equal_314936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 187)
        assert_array_almost_equal_call_result_314958 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_array_almost_equal_314936, *[subscript_call_result_314943, subscript_call_result_314956], **kwargs_314957)
        
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to array(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_314961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        float_314962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314962)
        # Adding element type (line 190)
        float_314963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314963)
        # Adding element type (line 190)
        float_314964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 43), 'float')
        # Getting the type of 'width' (line 190)
        width_314965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 47), 'width', False)
        int_314966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 53), 'int')
        # Applying the binary operator 'div' (line 190)
        result_div_314967 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 47), 'div', width_314965, int_314966)
        
        # Applying the binary operator '-' (line 190)
        result_sub_314968 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 43), '-', float_314964, result_div_314967)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_sub_314968)
        # Adding element type (line 190)
        float_314969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 56), 'float')
        # Getting the type of 'width' (line 190)
        width_314970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 60), 'width', False)
        int_314971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 66), 'int')
        # Applying the binary operator 'div' (line 190)
        result_div_314972 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 60), 'div', width_314970, int_314971)
        
        # Applying the binary operator '+' (line 190)
        result_add_314973 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 56), '+', float_314969, result_div_314972)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_add_314973)
        # Adding element type (line 190)
        float_314974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314974)
        # Adding element type (line 190)
        float_314975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 32), 'float')
        # Getting the type of 'width' (line 191)
        width_314976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'width', False)
        int_314977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 42), 'int')
        # Applying the binary operator 'div' (line 191)
        result_div_314978 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 36), 'div', width_314976, int_314977)
        
        # Applying the binary operator '-' (line 191)
        result_sub_314979 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 32), '-', float_314975, result_div_314978)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_sub_314979)
        # Adding element type (line 190)
        float_314980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 45), 'float')
        # Getting the type of 'width' (line 191)
        width_314981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 49), 'width', False)
        int_314982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 55), 'int')
        # Applying the binary operator 'div' (line 191)
        result_div_314983 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 49), 'div', width_314981, int_314982)
        
        # Applying the binary operator '+' (line 191)
        result_add_314984 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 45), '+', float_314980, result_div_314983)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_add_314984)
        # Adding element type (line 190)
        float_314985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314985)
        # Adding element type (line 190)
        float_314986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 32), 'float')
        # Getting the type of 'width' (line 192)
        width_314987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 36), 'width', False)
        int_314988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 42), 'int')
        # Applying the binary operator 'div' (line 192)
        result_div_314989 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 36), 'div', width_314987, int_314988)
        
        # Applying the binary operator '-' (line 192)
        result_sub_314990 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 32), '-', float_314986, result_div_314989)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_sub_314990)
        # Adding element type (line 190)
        float_314991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 45), 'float')
        # Getting the type of 'width' (line 192)
        width_314992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 49), 'width', False)
        int_314993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 55), 'int')
        # Applying the binary operator 'div' (line 192)
        result_div_314994 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 49), 'div', width_314992, int_314993)
        
        # Applying the binary operator '+' (line 192)
        result_add_314995 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 45), '+', float_314991, result_div_314994)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, result_add_314995)
        # Adding element type (line 190)
        float_314996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314996)
        # Adding element type (line 190)
        float_314997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 32), list_314961, float_314997)
        
        # Processing the call keyword arguments (line 190)
        kwargs_314998 = {}
        # Getting the type of 'np' (line 190)
        np_314959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 190)
        array_314960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 23), np_314959, 'array')
        # Calling array(args, kwargs) (line 190)
        array_call_result_314999 = invoke(stypy.reporting.localization.Localization(__file__, 190, 23), array_314960, *[list_314961], **kwargs_314998)
        
        # Assigning a type to the variable 'freq_samples' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'freq_samples', array_call_result_314999)
        
        # Assigning a Call to a Tuple (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_315000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        
        # Call to freqz(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'taps' (line 193)
        taps_315002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'taps', False)
        # Processing the call keyword arguments (line 193)
        # Getting the type of 'np' (line 193)
        np_315003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 193)
        pi_315004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 43), np_315003, 'pi')
        # Getting the type of 'freq_samples' (line 193)
        freq_samples_315005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 193)
        result_mul_315006 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 43), '*', pi_315004, freq_samples_315005)
        
        keyword_315007 = result_mul_315006
        kwargs_315008 = {'worN': keyword_315007}
        # Getting the type of 'freqz' (line 193)
        freqz_315001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 193)
        freqz_call_result_315009 = invoke(stypy.reporting.localization.Localization(__file__, 193, 26), freqz_315001, *[taps_315002], **kwargs_315008)
        
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___315010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), freqz_call_result_315009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_315011 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___315010, int_315000)
        
        # Assigning a type to the variable 'tuple_var_assignment_313923' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_313923', subscript_call_result_315011)
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_315012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        
        # Call to freqz(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'taps' (line 193)
        taps_315014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'taps', False)
        # Processing the call keyword arguments (line 193)
        # Getting the type of 'np' (line 193)
        np_315015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 193)
        pi_315016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 43), np_315015, 'pi')
        # Getting the type of 'freq_samples' (line 193)
        freq_samples_315017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 193)
        result_mul_315018 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 43), '*', pi_315016, freq_samples_315017)
        
        keyword_315019 = result_mul_315018
        kwargs_315020 = {'worN': keyword_315019}
        # Getting the type of 'freqz' (line 193)
        freqz_315013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 193)
        freqz_call_result_315021 = invoke(stypy.reporting.localization.Localization(__file__, 193, 26), freqz_315013, *[taps_315014], **kwargs_315020)
        
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___315022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), freqz_call_result_315021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_315023 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___315022, int_315012)
        
        # Assigning a type to the variable 'tuple_var_assignment_313924' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_313924', subscript_call_result_315023)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_313923' (line 193)
        tuple_var_assignment_313923_315024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_313923')
        # Assigning a type to the variable 'freqs' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'freqs', tuple_var_assignment_313923_315024)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_313924' (line 193)
        tuple_var_assignment_313924_315025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_313924')
        # Assigning a type to the variable 'response' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'response', tuple_var_assignment_313924_315025)
        
        # Call to assert_array_almost_equal(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Call to abs(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'response' (line 194)
        response_315029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 41), 'response', False)
        # Processing the call keyword arguments (line 194)
        kwargs_315030 = {}
        # Getting the type of 'np' (line 194)
        np_315027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 194)
        abs_315028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 34), np_315027, 'abs')
        # Calling abs(args, kwargs) (line 194)
        abs_call_result_315031 = invoke(stypy.reporting.localization.Localization(__file__, 194, 34), abs_315028, *[response_315029], **kwargs_315030)
        
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_315032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        float_315033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315033)
        # Adding element type (line 195)
        float_315034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315034)
        # Adding element type (line 195)
        float_315035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315035)
        # Adding element type (line 195)
        float_315036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315036)
        # Adding element type (line 195)
        float_315037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315037)
        # Adding element type (line 195)
        float_315038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315038)
        # Adding element type (line 195)
        float_315039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315039)
        # Adding element type (line 195)
        float_315040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315040)
        # Adding element type (line 195)
        float_315041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315041)
        # Adding element type (line 195)
        float_315042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315042)
        # Adding element type (line 195)
        float_315043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315043)
        # Adding element type (line 195)
        float_315044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 16), list_315032, float_315044)
        
        # Processing the call keyword arguments (line 194)
        int_315045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 24), 'int')
        keyword_315046 = int_315045
        kwargs_315047 = {'decimal': keyword_315046}
        # Getting the type of 'assert_array_almost_equal' (line 194)
        assert_array_almost_equal_315026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 194)
        assert_array_almost_equal_call_result_315048 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assert_array_almost_equal_315026, *[abs_call_result_315031, list_315032], **kwargs_315047)
        
        
        # ################# End of 'test_multi(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multi' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_315049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multi'
        return stypy_return_type_315049


    @norecursion
    def test_fs_nyq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fs_nyq'
        module_type_store = module_type_store.open_function_context('test_fs_nyq', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_fs_nyq')
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_fs_nyq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_fs_nyq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fs_nyq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fs_nyq(...)' code ##################

        str_315050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'str', 'Test the fs and nyq keywords.')
        
        # Assigning a Num to a Name (line 200):
        
        # Assigning a Num to a Name (line 200):
        int_315051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'int')
        # Assigning a type to the variable 'nyquist' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'nyquist', int_315051)
        
        # Assigning a Num to a Name (line 201):
        
        # Assigning a Num to a Name (line 201):
        float_315052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'float')
        # Assigning a type to the variable 'width' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'width', float_315052)
        
        # Assigning a BinOp to a Name (line 202):
        
        # Assigning a BinOp to a Name (line 202):
        # Getting the type of 'width' (line 202)
        width_315053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'width')
        # Getting the type of 'nyquist' (line 202)
        nyquist_315054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'nyquist')
        # Applying the binary operator 'div' (line 202)
        result_div_315055 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 25), 'div', width_315053, nyquist_315054)
        
        # Assigning a type to the variable 'relative_width' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'relative_width', result_div_315055)
        
        # Assigning a Call to a Tuple (line 203):
        
        # Assigning a Subscript to a Name (line 203):
        
        # Obtaining the type of the subscript
        int_315056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 8), 'int')
        
        # Call to kaiserord(...): (line 203)
        # Processing the call arguments (line 203)
        int_315058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'int')
        # Getting the type of 'relative_width' (line 203)
        relative_width_315059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'relative_width', False)
        # Processing the call keyword arguments (line 203)
        kwargs_315060 = {}
        # Getting the type of 'kaiserord' (line 203)
        kaiserord_315057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 203)
        kaiserord_call_result_315061 = invoke(stypy.reporting.localization.Localization(__file__, 203, 22), kaiserord_315057, *[int_315058, relative_width_315059], **kwargs_315060)
        
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___315062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), kaiserord_call_result_315061, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_315063 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), getitem___315062, int_315056)
        
        # Assigning a type to the variable 'tuple_var_assignment_313925' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_313925', subscript_call_result_315063)
        
        # Assigning a Subscript to a Name (line 203):
        
        # Obtaining the type of the subscript
        int_315064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 8), 'int')
        
        # Call to kaiserord(...): (line 203)
        # Processing the call arguments (line 203)
        int_315066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'int')
        # Getting the type of 'relative_width' (line 203)
        relative_width_315067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 37), 'relative_width', False)
        # Processing the call keyword arguments (line 203)
        kwargs_315068 = {}
        # Getting the type of 'kaiserord' (line 203)
        kaiserord_315065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 203)
        kaiserord_call_result_315069 = invoke(stypy.reporting.localization.Localization(__file__, 203, 22), kaiserord_315065, *[int_315066, relative_width_315067], **kwargs_315068)
        
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___315070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), kaiserord_call_result_315069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_315071 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), getitem___315070, int_315064)
        
        # Assigning a type to the variable 'tuple_var_assignment_313926' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_313926', subscript_call_result_315071)
        
        # Assigning a Name to a Name (line 203):
        # Getting the type of 'tuple_var_assignment_313925' (line 203)
        tuple_var_assignment_313925_315072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_313925')
        # Assigning a type to the variable 'ntaps' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'ntaps', tuple_var_assignment_313925_315072)
        
        # Assigning a Name to a Name (line 203):
        # Getting the type of 'tuple_var_assignment_313926' (line 203)
        tuple_var_assignment_313926_315073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tuple_var_assignment_313926')
        # Assigning a type to the variable 'beta' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'beta', tuple_var_assignment_313926_315073)
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to firwin(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'ntaps' (line 204)
        ntaps_315075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'ntaps', False)
        # Processing the call keyword arguments (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_315076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        int_315077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 36), list_315076, int_315077)
        # Adding element type (line 204)
        int_315078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 36), list_315076, int_315078)
        
        keyword_315079 = list_315076
        
        # Obtaining an instance of the builtin type 'tuple' (line 204)
        tuple_315080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 204)
        # Adding element type (line 204)
        str_315081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 56), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 56), tuple_315080, str_315081)
        # Adding element type (line 204)
        # Getting the type of 'beta' (line 204)
        beta_315082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 66), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 56), tuple_315080, beta_315082)
        
        keyword_315083 = tuple_315080
        # Getting the type of 'False' (line 205)
        False_315084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 34), 'False', False)
        keyword_315085 = False_315084
        # Getting the type of 'False' (line 205)
        False_315086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 47), 'False', False)
        keyword_315087 = False_315086
        int_315088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 57), 'int')
        # Getting the type of 'nyquist' (line 205)
        nyquist_315089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'nyquist', False)
        # Applying the binary operator '*' (line 205)
        result_mul_315090 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 57), '*', int_315088, nyquist_315089)
        
        keyword_315091 = result_mul_315090
        kwargs_315092 = {'cutoff': keyword_315079, 'window': keyword_315083, 'scale': keyword_315087, 'fs': keyword_315091, 'pass_zero': keyword_315085}
        # Getting the type of 'firwin' (line 204)
        firwin_315074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'firwin', False)
        # Calling firwin(args, kwargs) (line 204)
        firwin_call_result_315093 = invoke(stypy.reporting.localization.Localization(__file__, 204, 15), firwin_315074, *[ntaps_315075], **kwargs_315092)
        
        # Assigning a type to the variable 'taps' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'taps', firwin_call_result_315093)
        
        # Call to assert_array_almost_equal(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 208)
        ntaps_315095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 40), 'ntaps', False)
        int_315096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 47), 'int')
        # Applying the binary operator '//' (line 208)
        result_floordiv_315097 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 40), '//', ntaps_315095, int_315096)
        
        slice_315098 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 208, 34), None, result_floordiv_315097, None)
        # Getting the type of 'taps' (line 208)
        taps_315099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___315100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 34), taps_315099, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_315101 = invoke(stypy.reporting.localization.Localization(__file__, 208, 34), getitem___315100, slice_315098)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 208)
        ntaps_315102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 56), 'ntaps', False)
        # Getting the type of 'ntaps' (line 208)
        ntaps_315103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 62), 'ntaps', False)
        # Getting the type of 'ntaps' (line 208)
        ntaps_315104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 68), 'ntaps', False)
        int_315105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 75), 'int')
        # Applying the binary operator '//' (line 208)
        result_floordiv_315106 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 68), '//', ntaps_315104, int_315105)
        
        # Applying the binary operator '-' (line 208)
        result_sub_315107 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 62), '-', ntaps_315103, result_floordiv_315106)
        
        int_315108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 77), 'int')
        # Applying the binary operator '-' (line 208)
        result_sub_315109 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 76), '-', result_sub_315107, int_315108)
        
        int_315110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 79), 'int')
        slice_315111 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 208, 51), ntaps_315102, result_sub_315109, int_315110)
        # Getting the type of 'taps' (line 208)
        taps_315112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___315113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 51), taps_315112, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_315114 = invoke(stypy.reporting.localization.Localization(__file__, 208, 51), getitem___315113, slice_315111)
        
        # Processing the call keyword arguments (line 208)
        kwargs_315115 = {}
        # Getting the type of 'assert_array_almost_equal' (line 208)
        assert_array_almost_equal_315094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 208)
        assert_array_almost_equal_call_result_315116 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assert_array_almost_equal_315094, *[subscript_call_result_315101, subscript_call_result_315114], **kwargs_315115)
        
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to array(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_315119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        float_315120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, float_315120)
        # Adding element type (line 211)
        int_315121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, int_315121)
        # Adding element type (line 211)
        int_315122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'int')
        # Getting the type of 'width' (line 211)
        width_315123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 47), 'width', False)
        int_315124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 53), 'int')
        # Applying the binary operator 'div' (line 211)
        result_div_315125 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 47), 'div', width_315123, int_315124)
        
        # Applying the binary operator '-' (line 211)
        result_sub_315126 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 43), '-', int_315122, result_div_315125)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, result_sub_315126)
        # Adding element type (line 211)
        int_315127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 56), 'int')
        # Getting the type of 'width' (line 211)
        width_315128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 60), 'width', False)
        int_315129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 66), 'int')
        # Applying the binary operator 'div' (line 211)
        result_div_315130 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 60), 'div', width_315128, int_315129)
        
        # Applying the binary operator '+' (line 211)
        result_add_315131 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 56), '+', int_315127, result_div_315130)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, result_add_315131)
        # Adding element type (line 211)
        int_315132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, int_315132)
        # Adding element type (line 211)
        int_315133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 32), 'int')
        # Getting the type of 'width' (line 212)
        width_315134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'width', False)
        int_315135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 42), 'int')
        # Applying the binary operator 'div' (line 212)
        result_div_315136 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 36), 'div', width_315134, int_315135)
        
        # Applying the binary operator '-' (line 212)
        result_sub_315137 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 32), '-', int_315133, result_div_315136)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, result_sub_315137)
        # Adding element type (line 211)
        int_315138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 45), 'int')
        # Getting the type of 'width' (line 212)
        width_315139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 49), 'width', False)
        int_315140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 55), 'int')
        # Applying the binary operator 'div' (line 212)
        result_div_315141 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 49), 'div', width_315139, int_315140)
        
        # Applying the binary operator '+' (line 212)
        result_add_315142 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 45), '+', int_315138, result_div_315141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, result_add_315142)
        # Adding element type (line 211)
        int_315143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, int_315143)
        # Adding element type (line 211)
        int_315144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 32), list_315119, int_315144)
        
        # Processing the call keyword arguments (line 211)
        kwargs_315145 = {}
        # Getting the type of 'np' (line 211)
        np_315117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 211)
        array_315118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 23), np_315117, 'array')
        # Calling array(args, kwargs) (line 211)
        array_call_result_315146 = invoke(stypy.reporting.localization.Localization(__file__, 211, 23), array_315118, *[list_315119], **kwargs_315145)
        
        # Assigning a type to the variable 'freq_samples' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'freq_samples', array_call_result_315146)
        
        # Assigning a Call to a Tuple (line 213):
        
        # Assigning a Subscript to a Name (line 213):
        
        # Obtaining the type of the subscript
        int_315147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 8), 'int')
        
        # Call to freqz(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'taps' (line 213)
        taps_315149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'taps', False)
        # Processing the call keyword arguments (line 213)
        # Getting the type of 'np' (line 213)
        np_315150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 213)
        pi_315151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 43), np_315150, 'pi')
        # Getting the type of 'freq_samples' (line 213)
        freq_samples_315152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 213)
        result_mul_315153 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 43), '*', pi_315151, freq_samples_315152)
        
        # Getting the type of 'nyquist' (line 213)
        nyquist_315154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 62), 'nyquist', False)
        # Applying the binary operator 'div' (line 213)
        result_div_315155 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 61), 'div', result_mul_315153, nyquist_315154)
        
        keyword_315156 = result_div_315155
        kwargs_315157 = {'worN': keyword_315156}
        # Getting the type of 'freqz' (line 213)
        freqz_315148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 213)
        freqz_call_result_315158 = invoke(stypy.reporting.localization.Localization(__file__, 213, 26), freqz_315148, *[taps_315149], **kwargs_315157)
        
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___315159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), freqz_call_result_315158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_315160 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), getitem___315159, int_315147)
        
        # Assigning a type to the variable 'tuple_var_assignment_313927' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tuple_var_assignment_313927', subscript_call_result_315160)
        
        # Assigning a Subscript to a Name (line 213):
        
        # Obtaining the type of the subscript
        int_315161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 8), 'int')
        
        # Call to freqz(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'taps' (line 213)
        taps_315163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'taps', False)
        # Processing the call keyword arguments (line 213)
        # Getting the type of 'np' (line 213)
        np_315164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 213)
        pi_315165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 43), np_315164, 'pi')
        # Getting the type of 'freq_samples' (line 213)
        freq_samples_315166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 213)
        result_mul_315167 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 43), '*', pi_315165, freq_samples_315166)
        
        # Getting the type of 'nyquist' (line 213)
        nyquist_315168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 62), 'nyquist', False)
        # Applying the binary operator 'div' (line 213)
        result_div_315169 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 61), 'div', result_mul_315167, nyquist_315168)
        
        keyword_315170 = result_div_315169
        kwargs_315171 = {'worN': keyword_315170}
        # Getting the type of 'freqz' (line 213)
        freqz_315162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 213)
        freqz_call_result_315172 = invoke(stypy.reporting.localization.Localization(__file__, 213, 26), freqz_315162, *[taps_315163], **kwargs_315171)
        
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___315173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), freqz_call_result_315172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_315174 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), getitem___315173, int_315161)
        
        # Assigning a type to the variable 'tuple_var_assignment_313928' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tuple_var_assignment_313928', subscript_call_result_315174)
        
        # Assigning a Name to a Name (line 213):
        # Getting the type of 'tuple_var_assignment_313927' (line 213)
        tuple_var_assignment_313927_315175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tuple_var_assignment_313927')
        # Assigning a type to the variable 'freqs' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'freqs', tuple_var_assignment_313927_315175)
        
        # Assigning a Name to a Name (line 213):
        # Getting the type of 'tuple_var_assignment_313928' (line 213)
        tuple_var_assignment_313928_315176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'tuple_var_assignment_313928')
        # Assigning a type to the variable 'response' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'response', tuple_var_assignment_313928_315176)
        
        # Call to assert_array_almost_equal(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to abs(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'response' (line 214)
        response_315180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 41), 'response', False)
        # Processing the call keyword arguments (line 214)
        kwargs_315181 = {}
        # Getting the type of 'np' (line 214)
        np_315178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 214)
        abs_315179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 34), np_315178, 'abs')
        # Calling abs(args, kwargs) (line 214)
        abs_call_result_315182 = invoke(stypy.reporting.localization.Localization(__file__, 214, 34), abs_315179, *[response_315180], **kwargs_315181)
        
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_315183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        float_315184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315184)
        # Adding element type (line 215)
        float_315185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315185)
        # Adding element type (line 215)
        float_315186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315186)
        # Adding element type (line 215)
        float_315187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315187)
        # Adding element type (line 215)
        float_315188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315188)
        # Adding element type (line 215)
        float_315189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315189)
        # Adding element type (line 215)
        float_315190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315190)
        # Adding element type (line 215)
        float_315191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315191)
        # Adding element type (line 215)
        float_315192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), list_315183, float_315192)
        
        # Processing the call keyword arguments (line 214)
        int_315193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 71), 'int')
        keyword_315194 = int_315193
        kwargs_315195 = {'decimal': keyword_315194}
        # Getting the type of 'assert_array_almost_equal' (line 214)
        assert_array_almost_equal_315177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 214)
        assert_array_almost_equal_call_result_315196 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_array_almost_equal_315177, *[abs_call_result_315182, list_315183], **kwargs_315195)
        
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to firwin(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'ntaps' (line 217)
        ntaps_315198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'ntaps', False)
        # Processing the call keyword arguments (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_315199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        int_315200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 37), list_315199, int_315200)
        # Adding element type (line 217)
        int_315201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 37), list_315199, int_315201)
        
        keyword_315202 = list_315199
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_315203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        str_315204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 57), tuple_315203, str_315204)
        # Adding element type (line 217)
        # Getting the type of 'beta' (line 217)
        beta_315205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 67), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 57), tuple_315203, beta_315205)
        
        keyword_315206 = tuple_315203
        # Getting the type of 'False' (line 218)
        False_315207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'False', False)
        keyword_315208 = False_315207
        # Getting the type of 'False' (line 218)
        False_315209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 47), 'False', False)
        keyword_315210 = False_315209
        # Getting the type of 'nyquist' (line 218)
        nyquist_315211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 58), 'nyquist', False)
        keyword_315212 = nyquist_315211
        kwargs_315213 = {'cutoff': keyword_315202, 'window': keyword_315206, 'scale': keyword_315210, 'nyq': keyword_315212, 'pass_zero': keyword_315208}
        # Getting the type of 'firwin' (line 217)
        firwin_315197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'firwin', False)
        # Calling firwin(args, kwargs) (line 217)
        firwin_call_result_315214 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), firwin_315197, *[ntaps_315198], **kwargs_315213)
        
        # Assigning a type to the variable 'taps2' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'taps2', firwin_call_result_315214)
        
        # Call to assert_allclose(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'taps2' (line 219)
        taps2_315216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'taps2', False)
        # Getting the type of 'taps' (line 219)
        taps_315217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'taps', False)
        # Processing the call keyword arguments (line 219)
        kwargs_315218 = {}
        # Getting the type of 'assert_allclose' (line 219)
        assert_allclose_315215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 219)
        assert_allclose_call_result_315219 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assert_allclose_315215, *[taps2_315216, taps_315217], **kwargs_315218)
        
        
        # ################# End of 'test_fs_nyq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fs_nyq' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_315220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fs_nyq'
        return stypy_return_type_315220


    @norecursion
    def test_bad_cutoff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_cutoff'
        module_type_store = module_type_store.open_function_context('test_bad_cutoff', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_bad_cutoff')
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_bad_cutoff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_bad_cutoff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_cutoff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_cutoff(...)' code ##################

        str_315221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 8), 'str', 'Test that invalid cutoff argument raises ValueError.')
        
        # Call to assert_raises(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'ValueError' (line 224)
        ValueError_315223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 224)
        firwin_315224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'firwin', False)
        int_315225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 42), 'int')
        float_315226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 46), 'float')
        # Processing the call keyword arguments (line 224)
        kwargs_315227 = {}
        # Getting the type of 'assert_raises' (line 224)
        assert_raises_315222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 224)
        assert_raises_call_result_315228 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), assert_raises_315222, *[ValueError_315223, firwin_315224, int_315225, float_315226], **kwargs_315227)
        
        
        # Call to assert_raises(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'ValueError' (line 225)
        ValueError_315230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 225)
        firwin_315231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 34), 'firwin', False)
        int_315232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 42), 'int')
        float_315233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 46), 'float')
        # Processing the call keyword arguments (line 225)
        kwargs_315234 = {}
        # Getting the type of 'assert_raises' (line 225)
        assert_raises_315229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 225)
        assert_raises_call_result_315235 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), assert_raises_315229, *[ValueError_315230, firwin_315231, int_315232, float_315233], **kwargs_315234)
        
        
        # Call to assert_raises(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'ValueError' (line 227)
        ValueError_315237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 227)
        firwin_315238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 34), 'firwin', False)
        int_315239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_315240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        int_315241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 46), list_315240, int_315241)
        # Adding element type (line 227)
        float_315242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 46), list_315240, float_315242)
        
        # Processing the call keyword arguments (line 227)
        kwargs_315243 = {}
        # Getting the type of 'assert_raises' (line 227)
        assert_raises_315236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 227)
        assert_raises_call_result_315244 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assert_raises_315236, *[ValueError_315237, firwin_315238, int_315239, list_315240], **kwargs_315243)
        
        
        # Call to assert_raises(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'ValueError' (line 228)
        ValueError_315246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 228)
        firwin_315247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'firwin', False)
        int_315248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_315249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        float_315250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 46), list_315249, float_315250)
        # Adding element type (line 228)
        int_315251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 46), list_315249, int_315251)
        
        # Processing the call keyword arguments (line 228)
        kwargs_315252 = {}
        # Getting the type of 'assert_raises' (line 228)
        assert_raises_315245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 228)
        assert_raises_call_result_315253 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert_raises_315245, *[ValueError_315246, firwin_315247, int_315248, list_315249], **kwargs_315252)
        
        
        # Call to assert_raises(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'ValueError' (line 230)
        ValueError_315255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 230)
        firwin_315256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 'firwin', False)
        int_315257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_315258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        float_315259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 46), list_315258, float_315259)
        # Adding element type (line 230)
        float_315260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 46), list_315258, float_315260)
        # Adding element type (line 230)
        float_315261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 46), list_315258, float_315261)
        
        # Processing the call keyword arguments (line 230)
        kwargs_315262 = {}
        # Getting the type of 'assert_raises' (line 230)
        assert_raises_315254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 230)
        assert_raises_call_result_315263 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert_raises_315254, *[ValueError_315255, firwin_315256, int_315257, list_315258], **kwargs_315262)
        
        
        # Call to assert_raises(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'ValueError' (line 231)
        ValueError_315265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 231)
        firwin_315266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'firwin', False)
        int_315267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_315268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        float_315269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 46), list_315268, float_315269)
        # Adding element type (line 231)
        float_315270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 46), list_315268, float_315270)
        # Adding element type (line 231)
        float_315271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 46), list_315268, float_315271)
        
        # Processing the call keyword arguments (line 231)
        kwargs_315272 = {}
        # Getting the type of 'assert_raises' (line 231)
        assert_raises_315264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 231)
        assert_raises_call_result_315273 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), assert_raises_315264, *[ValueError_315265, firwin_315266, int_315267, list_315268], **kwargs_315272)
        
        
        # Call to assert_raises(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'ValueError' (line 233)
        ValueError_315275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 233)
        firwin_315276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 34), 'firwin', False)
        int_315277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_315278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        
        # Processing the call keyword arguments (line 233)
        kwargs_315279 = {}
        # Getting the type of 'assert_raises' (line 233)
        assert_raises_315274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 233)
        assert_raises_call_result_315280 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert_raises_315274, *[ValueError_315275, firwin_315276, int_315277, list_315278], **kwargs_315279)
        
        
        # Call to assert_raises(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'ValueError' (line 235)
        ValueError_315282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 235)
        firwin_315283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'firwin', False)
        int_315284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_315285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_315286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        float_315287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 47), list_315286, float_315287)
        # Adding element type (line 235)
        float_315288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 47), list_315286, float_315288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 46), list_315285, list_315286)
        # Adding element type (line 235)
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_315289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        float_315290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 58), list_315289, float_315290)
        # Adding element type (line 235)
        float_315291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 58), list_315289, float_315291)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 46), list_315285, list_315289)
        
        # Processing the call keyword arguments (line 235)
        kwargs_315292 = {}
        # Getting the type of 'assert_raises' (line 235)
        assert_raises_315281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 235)
        assert_raises_call_result_315293 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), assert_raises_315281, *[ValueError_315282, firwin_315283, int_315284, list_315285], **kwargs_315292)
        
        
        # Call to assert_raises(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'ValueError' (line 237)
        ValueError_315295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 237)
        firwin_315296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'firwin', False)
        int_315297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 42), 'int')
        float_315298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 46), 'float')
        # Processing the call keyword arguments (line 237)
        int_315299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 56), 'int')
        keyword_315300 = int_315299
        kwargs_315301 = {'nyq': keyword_315300}
        # Getting the type of 'assert_raises' (line 237)
        assert_raises_315294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 237)
        assert_raises_call_result_315302 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assert_raises_315294, *[ValueError_315295, firwin_315296, int_315297, float_315298], **kwargs_315301)
        
        
        # Call to assert_raises(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'ValueError' (line 238)
        ValueError_315304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 238)
        firwin_315305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 34), 'firwin', False)
        int_315306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_315307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        int_315308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 46), list_315307, int_315308)
        # Adding element type (line 238)
        int_315309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 46), list_315307, int_315309)
        # Adding element type (line 238)
        int_315310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 46), list_315307, int_315310)
        
        # Processing the call keyword arguments (line 238)
        int_315311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 64), 'int')
        keyword_315312 = int_315311
        kwargs_315313 = {'nyq': keyword_315312}
        # Getting the type of 'assert_raises' (line 238)
        assert_raises_315303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 238)
        assert_raises_call_result_315314 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assert_raises_315303, *[ValueError_315304, firwin_315305, int_315306, list_315307], **kwargs_315313)
        
        
        # Call to assert_raises(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'ValueError' (line 239)
        ValueError_315316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 239)
        firwin_315317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 'firwin', False)
        int_315318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 42), 'int')
        float_315319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 46), 'float')
        # Processing the call keyword arguments (line 239)
        int_315320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 55), 'int')
        keyword_315321 = int_315320
        kwargs_315322 = {'fs': keyword_315321}
        # Getting the type of 'assert_raises' (line 239)
        assert_raises_315315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 239)
        assert_raises_call_result_315323 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), assert_raises_315315, *[ValueError_315316, firwin_315317, int_315318, float_315319], **kwargs_315322)
        
        
        # Call to assert_raises(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'ValueError' (line 240)
        ValueError_315325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 240)
        firwin_315326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'firwin', False)
        int_315327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_315328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        # Adding element type (line 240)
        int_315329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 46), list_315328, int_315329)
        # Adding element type (line 240)
        int_315330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 46), list_315328, int_315330)
        # Adding element type (line 240)
        int_315331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 46), list_315328, int_315331)
        
        # Processing the call keyword arguments (line 240)
        int_315332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 63), 'int')
        keyword_315333 = int_315332
        kwargs_315334 = {'fs': keyword_315333}
        # Getting the type of 'assert_raises' (line 240)
        assert_raises_315324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 240)
        assert_raises_call_result_315335 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assert_raises_315324, *[ValueError_315325, firwin_315326, int_315327, list_315328], **kwargs_315334)
        
        
        # ################# End of 'test_bad_cutoff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_cutoff' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_315336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_cutoff'
        return stypy_return_type_315336


    @norecursion
    def test_even_highpass_raises_value_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_even_highpass_raises_value_error'
        module_type_store = module_type_store.open_function_context('test_even_highpass_raises_value_error', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_localization', localization)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_function_name', 'TestFirWinMore.test_even_highpass_raises_value_error')
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirWinMore.test_even_highpass_raises_value_error.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.test_even_highpass_raises_value_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_even_highpass_raises_value_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_even_highpass_raises_value_error(...)' code ##################

        str_315337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', 'Test that attempt to create a highpass filter with an even number\n        of taps raises a ValueError exception.')
        
        # Call to assert_raises(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'ValueError' (line 245)
        ValueError_315339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 245)
        firwin_315340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'firwin', False)
        int_315341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 42), 'int')
        float_315342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 46), 'float')
        # Processing the call keyword arguments (line 245)
        # Getting the type of 'False' (line 245)
        False_315343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 61), 'False', False)
        keyword_315344 = False_315343
        kwargs_315345 = {'pass_zero': keyword_315344}
        # Getting the type of 'assert_raises' (line 245)
        assert_raises_315338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 245)
        assert_raises_call_result_315346 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_raises_315338, *[ValueError_315339, firwin_315340, int_315341, float_315342], **kwargs_315345)
        
        
        # Call to assert_raises(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'ValueError' (line 246)
        ValueError_315348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'ValueError', False)
        # Getting the type of 'firwin' (line 246)
        firwin_315349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'firwin', False)
        int_315350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 42), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_315351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        float_315352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 46), list_315351, float_315352)
        # Adding element type (line 246)
        float_315353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 46), list_315351, float_315353)
        
        # Processing the call keyword arguments (line 246)
        kwargs_315354 = {}
        # Getting the type of 'assert_raises' (line 246)
        assert_raises_315347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 246)
        assert_raises_call_result_315355 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), assert_raises_315347, *[ValueError_315348, firwin_315349, int_315350, list_315351], **kwargs_315354)
        
        
        # ################# End of 'test_even_highpass_raises_value_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_even_highpass_raises_value_error' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_315356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315356)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_even_highpass_raises_value_error'
        return stypy_return_type_315356


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 128, 0, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirWinMore.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFirWinMore' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'TestFirWinMore', TestFirWinMore)
# Declaration of the 'TestFirwin2' class

class TestFirwin2(object, ):

    @norecursion
    def test_invalid_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_invalid_args'
        module_type_store = module_type_store.open_function_context('test_invalid_args', 251, 4, False)
        # Assigning a type to the variable 'self' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test_invalid_args')
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test_invalid_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test_invalid_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_invalid_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_invalid_args(...)' code ##################

        
        # Call to assert_raises(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'ValueError' (line 253)
        ValueError_315358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 253)
        firwin2_315359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 34), 'firwin2', False)
        int_315360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_315361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        int_315362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 47), list_315361, int_315362)
        # Adding element type (line 253)
        float_315363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 47), list_315361, float_315363)
        # Adding element type (line 253)
        int_315364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 47), list_315361, int_315364)
        
        
        # Obtaining an instance of the builtin type 'list' (line 253)
        list_315365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 253)
        # Adding element type (line 253)
        float_315366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 60), list_315365, float_315366)
        # Adding element type (line 253)
        float_315367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 60), list_315365, float_315367)
        
        # Processing the call keyword arguments (line 253)
        kwargs_315368 = {}
        # Getting the type of 'assert_raises' (line 253)
        assert_raises_315357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 253)
        assert_raises_call_result_315369 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), assert_raises_315357, *[ValueError_315358, firwin2_315359, int_315360, list_315361, list_315365], **kwargs_315368)
        
        
        # Call to assert_raises(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'ValueError' (line 255)
        ValueError_315371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 255)
        firwin2_315372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 34), 'firwin2', False)
        int_315373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_315374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        int_315375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 47), list_315374, int_315375)
        # Adding element type (line 255)
        float_315376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 47), list_315374, float_315376)
        # Adding element type (line 255)
        int_315377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 47), list_315374, int_315377)
        
        
        # Obtaining an instance of the builtin type 'list' (line 255)
        list_315378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 255)
        # Adding element type (line 255)
        float_315379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 60), list_315378, float_315379)
        # Adding element type (line 255)
        float_315380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 60), list_315378, float_315380)
        # Adding element type (line 255)
        float_315381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 60), list_315378, float_315381)
        
        # Processing the call keyword arguments (line 255)
        int_315382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 84), 'int')
        keyword_315383 = int_315382
        kwargs_315384 = {'nfreqs': keyword_315383}
        # Getting the type of 'assert_raises' (line 255)
        assert_raises_315370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 255)
        assert_raises_call_result_315385 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assert_raises_315370, *[ValueError_315371, firwin2_315372, int_315373, list_315374, list_315378], **kwargs_315384)
        
        
        # Call to assert_raises(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'ValueError' (line 257)
        ValueError_315387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 257)
        firwin2_315388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'firwin2', False)
        int_315389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_315390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        int_315391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 47), list_315390, int_315391)
        # Adding element type (line 257)
        float_315392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 47), list_315390, float_315392)
        # Adding element type (line 257)
        float_315393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 47), list_315390, float_315393)
        # Adding element type (line 257)
        float_315394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 47), list_315390, float_315394)
        
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_315395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        int_315396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 67), list_315395, int_315396)
        # Adding element type (line 257)
        float_315397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 67), list_315395, float_315397)
        # Adding element type (line 257)
        float_315398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 76), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 67), list_315395, float_315398)
        # Adding element type (line 257)
        float_315399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 80), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 67), list_315395, float_315399)
        
        # Processing the call keyword arguments (line 257)
        kwargs_315400 = {}
        # Getting the type of 'assert_raises' (line 257)
        assert_raises_315386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 257)
        assert_raises_call_result_315401 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), assert_raises_315386, *[ValueError_315387, firwin2_315388, int_315389, list_315390, list_315395], **kwargs_315400)
        
        
        # Call to assert_raises(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'ValueError' (line 259)
        ValueError_315403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 259)
        firwin2_315404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'firwin2', False)
        int_315405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_315406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        int_315407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 47), list_315406, int_315407)
        # Adding element type (line 259)
        float_315408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 47), list_315406, float_315408)
        # Adding element type (line 259)
        float_315409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 47), list_315406, float_315409)
        # Adding element type (line 259)
        float_315410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 47), list_315406, float_315410)
        # Adding element type (line 259)
        float_315411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 47), list_315406, float_315411)
        
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_315412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        float_315413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), list_315412, float_315413)
        # Adding element type (line 260)
        float_315414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), list_315412, float_315414)
        # Adding element type (line 260)
        float_315415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), list_315412, float_315415)
        # Adding element type (line 260)
        float_315416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), list_315412, float_315416)
        # Adding element type (line 260)
        float_315417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 47), list_315412, float_315417)
        
        # Processing the call keyword arguments (line 259)
        kwargs_315418 = {}
        # Getting the type of 'assert_raises' (line 259)
        assert_raises_315402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 259)
        assert_raises_call_result_315419 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), assert_raises_315402, *[ValueError_315403, firwin2_315404, int_315405, list_315406, list_315412], **kwargs_315418)
        
        
        # Call to assert_raises(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'ValueError' (line 262)
        ValueError_315421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 262)
        firwin2_315422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 34), 'firwin2', False)
        int_315423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_315424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        float_315425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 47), list_315424, float_315425)
        # Adding element type (line 262)
        float_315426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 47), list_315424, float_315426)
        
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_315427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        float_315428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 59), list_315427, float_315428)
        # Adding element type (line 262)
        float_315429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 59), list_315427, float_315429)
        
        # Processing the call keyword arguments (line 262)
        kwargs_315430 = {}
        # Getting the type of 'assert_raises' (line 262)
        assert_raises_315420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 262)
        assert_raises_call_result_315431 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_raises_315420, *[ValueError_315421, firwin2_315422, int_315423, list_315424, list_315427], **kwargs_315430)
        
        
        # Call to assert_raises(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'ValueError' (line 265)
        ValueError_315433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 265)
        firwin2_315434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 34), 'firwin2', False)
        int_315435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_315436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        # Adding element type (line 265)
        float_315437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 47), list_315436, float_315437)
        # Adding element type (line 265)
        float_315438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 47), list_315436, float_315438)
        # Adding element type (line 265)
        float_315439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 47), list_315436, float_315439)
        
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_315440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        # Adding element type (line 265)
        float_315441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 64), list_315440, float_315441)
        # Adding element type (line 265)
        float_315442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 64), list_315440, float_315442)
        # Adding element type (line 265)
        float_315443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 64), list_315440, float_315443)
        
        # Processing the call keyword arguments (line 265)
        kwargs_315444 = {}
        # Getting the type of 'assert_raises' (line 265)
        assert_raises_315432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 265)
        assert_raises_call_result_315445 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert_raises_315432, *[ValueError_315433, firwin2_315434, int_315435, list_315436, list_315440], **kwargs_315444)
        
        
        # Call to assert_raises(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'ValueError' (line 268)
        ValueError_315447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 268)
        firwin2_315448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 34), 'firwin2', False)
        int_315449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_315450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_315451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 47), list_315450, float_315451)
        # Adding element type (line 268)
        float_315452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 47), list_315450, float_315452)
        # Adding element type (line 268)
        float_315453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 47), list_315450, float_315453)
        
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_315454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        float_315455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 64), list_315454, float_315455)
        # Adding element type (line 268)
        float_315456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 64), list_315454, float_315456)
        # Adding element type (line 268)
        float_315457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 64), list_315454, float_315457)
        
        # Processing the call keyword arguments (line 268)
        # Getting the type of 'True' (line 269)
        True_315458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 36), 'True', False)
        keyword_315459 = True_315458
        kwargs_315460 = {'antisymmetric': keyword_315459}
        # Getting the type of 'assert_raises' (line 268)
        assert_raises_315446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 268)
        assert_raises_call_result_315461 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assert_raises_315446, *[ValueError_315447, firwin2_315448, int_315449, list_315450, list_315454], **kwargs_315460)
        
        
        # Call to assert_raises(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'ValueError' (line 270)
        ValueError_315463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 270)
        firwin2_315464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'firwin2', False)
        int_315465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_315466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        # Adding element type (line 270)
        float_315467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 47), list_315466, float_315467)
        # Adding element type (line 270)
        float_315468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 47), list_315466, float_315468)
        # Adding element type (line 270)
        float_315469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 47), list_315466, float_315469)
        
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_315470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        # Adding element type (line 270)
        float_315471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 64), list_315470, float_315471)
        # Adding element type (line 270)
        float_315472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 64), list_315470, float_315472)
        # Adding element type (line 270)
        float_315473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 64), list_315470, float_315473)
        
        # Processing the call keyword arguments (line 270)
        # Getting the type of 'True' (line 271)
        True_315474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 36), 'True', False)
        keyword_315475 = True_315474
        kwargs_315476 = {'antisymmetric': keyword_315475}
        # Getting the type of 'assert_raises' (line 270)
        assert_raises_315462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 270)
        assert_raises_call_result_315477 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), assert_raises_315462, *[ValueError_315463, firwin2_315464, int_315465, list_315466, list_315470], **kwargs_315476)
        
        
        # Call to assert_raises(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'ValueError' (line 272)
        ValueError_315479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 272)
        firwin2_315480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'firwin2', False)
        int_315481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_315482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        float_315483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 47), list_315482, float_315483)
        # Adding element type (line 272)
        float_315484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 47), list_315482, float_315484)
        # Adding element type (line 272)
        float_315485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 47), list_315482, float_315485)
        
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_315486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        # Adding element type (line 272)
        float_315487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 64), list_315486, float_315487)
        # Adding element type (line 272)
        float_315488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 64), list_315486, float_315488)
        # Adding element type (line 272)
        float_315489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 64), list_315486, float_315489)
        
        # Processing the call keyword arguments (line 272)
        # Getting the type of 'True' (line 273)
        True_315490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'True', False)
        keyword_315491 = True_315490
        kwargs_315492 = {'antisymmetric': keyword_315491}
        # Getting the type of 'assert_raises' (line 272)
        assert_raises_315478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 272)
        assert_raises_call_result_315493 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), assert_raises_315478, *[ValueError_315479, firwin2_315480, int_315481, list_315482, list_315486], **kwargs_315492)
        
        
        # Call to assert_raises(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'ValueError' (line 276)
        ValueError_315495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'ValueError', False)
        # Getting the type of 'firwin2' (line 276)
        firwin2_315496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'firwin2', False)
        int_315497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 43), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_315498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        float_315499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 47), list_315498, float_315499)
        # Adding element type (line 276)
        float_315500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 47), list_315498, float_315500)
        # Adding element type (line 276)
        float_315501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 47), list_315498, float_315501)
        
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_315502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        float_315503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 64), list_315502, float_315503)
        # Adding element type (line 276)
        float_315504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 64), list_315502, float_315504)
        # Adding element type (line 276)
        float_315505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 75), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 64), list_315502, float_315505)
        
        # Processing the call keyword arguments (line 276)
        # Getting the type of 'True' (line 277)
        True_315506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 36), 'True', False)
        keyword_315507 = True_315506
        kwargs_315508 = {'antisymmetric': keyword_315507}
        # Getting the type of 'assert_raises' (line 276)
        assert_raises_315494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 276)
        assert_raises_call_result_315509 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assert_raises_315494, *[ValueError_315495, firwin2_315496, int_315497, list_315498, list_315502], **kwargs_315508)
        
        
        # ################# End of 'test_invalid_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_invalid_args' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_315510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_invalid_args'
        return stypy_return_type_315510


    @norecursion
    def test01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test01'
        module_type_store = module_type_store.open_function_context('test01', 279, 4, False)
        # Assigning a type to the variable 'self' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test01.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test01.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test01')
        TestFirwin2.test01.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test01(...)' code ##################

        
        # Assigning a Num to a Name (line 280):
        
        # Assigning a Num to a Name (line 280):
        float_315511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'float')
        # Assigning a type to the variable 'width' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'width', float_315511)
        
        # Assigning a Num to a Name (line 281):
        
        # Assigning a Num to a Name (line 281):
        float_315512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 15), 'float')
        # Assigning a type to the variable 'beta' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'beta', float_315512)
        
        # Assigning a Num to a Name (line 282):
        
        # Assigning a Num to a Name (line 282):
        int_315513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 16), 'int')
        # Assigning a type to the variable 'ntaps' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'ntaps', int_315513)
        
        # Assigning a List to a Name (line 285):
        
        # Assigning a List to a Name (line 285):
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_315514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        float_315515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), list_315514, float_315515)
        # Adding element type (line 285)
        float_315516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), list_315514, float_315516)
        # Adding element type (line 285)
        float_315517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), list_315514, float_315517)
        
        # Assigning a type to the variable 'freq' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'freq', list_315514)
        
        # Assigning a List to a Name (line 286):
        
        # Assigning a List to a Name (line 286):
        
        # Obtaining an instance of the builtin type 'list' (line 286)
        list_315518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 286)
        # Adding element type (line 286)
        float_315519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 15), list_315518, float_315519)
        # Adding element type (line 286)
        float_315520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 15), list_315518, float_315520)
        # Adding element type (line 286)
        float_315521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 15), list_315518, float_315521)
        
        # Assigning a type to the variable 'gain' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'gain', list_315518)
        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to firwin2(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'ntaps' (line 287)
        ntaps_315523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 287)
        freq_315524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'freq', False)
        # Getting the type of 'gain' (line 287)
        gain_315525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'gain', False)
        # Processing the call keyword arguments (line 287)
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_315526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        str_315527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 50), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 50), tuple_315526, str_315527)
        # Adding element type (line 287)
        # Getting the type of 'beta' (line 287)
        beta_315528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 50), tuple_315526, beta_315528)
        
        keyword_315529 = tuple_315526
        kwargs_315530 = {'window': keyword_315529}
        # Getting the type of 'firwin2' (line 287)
        firwin2_315522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 287)
        firwin2_call_result_315531 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), firwin2_315522, *[ntaps_315523, freq_315524, gain_315525], **kwargs_315530)
        
        # Assigning a type to the variable 'taps' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'taps', firwin2_call_result_315531)
        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to array(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Obtaining an instance of the builtin type 'list' (line 288)
        list_315534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 288)
        # Adding element type (line 288)
        float_315535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, float_315535)
        # Adding element type (line 288)
        float_315536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, float_315536)
        # Adding element type (line 288)
        float_315537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 44), 'float')
        # Getting the type of 'width' (line 288)
        width_315538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 48), 'width', False)
        int_315539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 54), 'int')
        # Applying the binary operator 'div' (line 288)
        result_div_315540 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 48), 'div', width_315538, int_315539)
        
        # Applying the binary operator '-' (line 288)
        result_sub_315541 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 44), '-', float_315537, result_div_315540)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, result_sub_315541)
        # Adding element type (line 288)
        float_315542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 57), 'float')
        # Getting the type of 'width' (line 288)
        width_315543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 61), 'width', False)
        int_315544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 67), 'int')
        # Applying the binary operator 'div' (line 288)
        result_div_315545 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 61), 'div', width_315543, int_315544)
        
        # Applying the binary operator '+' (line 288)
        result_add_315546 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 57), '+', float_315542, result_div_315545)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, result_add_315546)
        # Adding element type (line 288)
        float_315547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, float_315547)
        # Adding element type (line 288)
        float_315548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 62), 'float')
        # Getting the type of 'width' (line 289)
        width_315549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 66), 'width', False)
        int_315550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 72), 'int')
        # Applying the binary operator 'div' (line 289)
        result_div_315551 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 66), 'div', width_315549, int_315550)
        
        # Applying the binary operator '-' (line 289)
        result_sub_315552 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 62), '-', float_315548, result_div_315551)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 32), list_315534, result_sub_315552)
        
        # Processing the call keyword arguments (line 288)
        kwargs_315553 = {}
        # Getting the type of 'np' (line 288)
        np_315532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 288)
        array_315533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 23), np_315532, 'array')
        # Calling array(args, kwargs) (line 288)
        array_call_result_315554 = invoke(stypy.reporting.localization.Localization(__file__, 288, 23), array_315533, *[list_315534], **kwargs_315553)
        
        # Assigning a type to the variable 'freq_samples' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'freq_samples', array_call_result_315554)
        
        # Assigning a Call to a Tuple (line 290):
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_315555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to freqz(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'taps' (line 290)
        taps_315557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'taps', False)
        # Processing the call keyword arguments (line 290)
        # Getting the type of 'np' (line 290)
        np_315558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 290)
        pi_315559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 43), np_315558, 'pi')
        # Getting the type of 'freq_samples' (line 290)
        freq_samples_315560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 290)
        result_mul_315561 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 43), '*', pi_315559, freq_samples_315560)
        
        keyword_315562 = result_mul_315561
        kwargs_315563 = {'worN': keyword_315562}
        # Getting the type of 'freqz' (line 290)
        freqz_315556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 290)
        freqz_call_result_315564 = invoke(stypy.reporting.localization.Localization(__file__, 290, 26), freqz_315556, *[taps_315557], **kwargs_315563)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___315565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), freqz_call_result_315564, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_315566 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___315565, int_315555)
        
        # Assigning a type to the variable 'tuple_var_assignment_313929' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_313929', subscript_call_result_315566)
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_315567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to freqz(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'taps' (line 290)
        taps_315569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'taps', False)
        # Processing the call keyword arguments (line 290)
        # Getting the type of 'np' (line 290)
        np_315570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 290)
        pi_315571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 43), np_315570, 'pi')
        # Getting the type of 'freq_samples' (line 290)
        freq_samples_315572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 290)
        result_mul_315573 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 43), '*', pi_315571, freq_samples_315572)
        
        keyword_315574 = result_mul_315573
        kwargs_315575 = {'worN': keyword_315574}
        # Getting the type of 'freqz' (line 290)
        freqz_315568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 290)
        freqz_call_result_315576 = invoke(stypy.reporting.localization.Localization(__file__, 290, 26), freqz_315568, *[taps_315569], **kwargs_315575)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___315577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), freqz_call_result_315576, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_315578 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___315577, int_315567)
        
        # Assigning a type to the variable 'tuple_var_assignment_313930' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_313930', subscript_call_result_315578)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_313929' (line 290)
        tuple_var_assignment_313929_315579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_313929')
        # Assigning a type to the variable 'freqs' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'freqs', tuple_var_assignment_313929_315579)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_313930' (line 290)
        tuple_var_assignment_313930_315580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_313930')
        # Assigning a type to the variable 'response' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), 'response', tuple_var_assignment_313930_315580)
        
        # Call to assert_array_almost_equal(...): (line 291)
        # Processing the call arguments (line 291)
        
        # Call to abs(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'response' (line 291)
        response_315584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'response', False)
        # Processing the call keyword arguments (line 291)
        kwargs_315585 = {}
        # Getting the type of 'np' (line 291)
        np_315582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 291)
        abs_315583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 34), np_315582, 'abs')
        # Calling abs(args, kwargs) (line 291)
        abs_call_result_315586 = invoke(stypy.reporting.localization.Localization(__file__, 291, 34), abs_315583, *[response_315584], **kwargs_315585)
        
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_315587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        float_315588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, float_315588)
        # Adding element type (line 292)
        float_315589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, float_315589)
        # Adding element type (line 292)
        float_315590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, float_315590)
        # Adding element type (line 292)
        float_315591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 40), 'float')
        # Getting the type of 'width' (line 292)
        width_315592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 44), 'width', False)
        # Applying the binary operator '-' (line 292)
        result_sub_315593 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 40), '-', float_315591, width_315592)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, result_sub_315593)
        # Adding element type (line 292)
        float_315594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, float_315594)
        # Adding element type (line 292)
        # Getting the type of 'width' (line 292)
        width_315595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 56), 'width', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 24), list_315587, width_315595)
        
        # Processing the call keyword arguments (line 291)
        int_315596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 72), 'int')
        keyword_315597 = int_315596
        kwargs_315598 = {'decimal': keyword_315597}
        # Getting the type of 'assert_array_almost_equal' (line 291)
        assert_array_almost_equal_315581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 291)
        assert_array_almost_equal_call_result_315599 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assert_array_almost_equal_315581, *[abs_call_result_315586, list_315587], **kwargs_315598)
        
        
        # ################# End of 'test01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test01' in the type store
        # Getting the type of 'stypy_return_type' (line 279)
        stypy_return_type_315600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test01'
        return stypy_return_type_315600


    @norecursion
    def test02(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test02'
        module_type_store = module_type_store.open_function_context('test02', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test02.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test02.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test02.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test02.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test02')
        TestFirwin2.test02.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test02.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test02.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test02.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test02.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test02.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test02.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test02', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test02', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test02(...)' code ##################

        
        # Assigning a Num to a Name (line 295):
        
        # Assigning a Num to a Name (line 295):
        float_315601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 16), 'float')
        # Assigning a type to the variable 'width' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'width', float_315601)
        
        # Assigning a Num to a Name (line 296):
        
        # Assigning a Num to a Name (line 296):
        float_315602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 15), 'float')
        # Assigning a type to the variable 'beta' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'beta', float_315602)
        
        # Assigning a Num to a Name (line 298):
        
        # Assigning a Num to a Name (line 298):
        int_315603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 16), 'int')
        # Assigning a type to the variable 'ntaps' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'ntaps', int_315603)
        
        # Assigning a List to a Name (line 300):
        
        # Assigning a List to a Name (line 300):
        
        # Obtaining an instance of the builtin type 'list' (line 300)
        list_315604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 300)
        # Adding element type (line 300)
        float_315605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), list_315604, float_315605)
        # Adding element type (line 300)
        float_315606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), list_315604, float_315606)
        # Adding element type (line 300)
        float_315607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), list_315604, float_315607)
        # Adding element type (line 300)
        float_315608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 15), list_315604, float_315608)
        
        # Assigning a type to the variable 'freq' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'freq', list_315604)
        
        # Assigning a List to a Name (line 301):
        
        # Assigning a List to a Name (line 301):
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_315609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        # Adding element type (line 301)
        float_315610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 15), list_315609, float_315610)
        # Adding element type (line 301)
        float_315611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 15), list_315609, float_315611)
        # Adding element type (line 301)
        float_315612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 15), list_315609, float_315612)
        # Adding element type (line 301)
        float_315613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 15), list_315609, float_315613)
        
        # Assigning a type to the variable 'gain' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'gain', list_315609)
        
        # Assigning a Call to a Name (line 302):
        
        # Assigning a Call to a Name (line 302):
        
        # Call to firwin2(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'ntaps' (line 302)
        ntaps_315615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 302)
        freq_315616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'freq', False)
        # Getting the type of 'gain' (line 302)
        gain_315617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 36), 'gain', False)
        # Processing the call keyword arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_315618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        str_315619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 50), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 50), tuple_315618, str_315619)
        # Adding element type (line 302)
        # Getting the type of 'beta' (line 302)
        beta_315620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 60), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 50), tuple_315618, beta_315620)
        
        keyword_315621 = tuple_315618
        kwargs_315622 = {'window': keyword_315621}
        # Getting the type of 'firwin2' (line 302)
        firwin2_315614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 302)
        firwin2_call_result_315623 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), firwin2_315614, *[ntaps_315615, freq_315616, gain_315617], **kwargs_315622)
        
        # Assigning a type to the variable 'taps' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'taps', firwin2_call_result_315623)
        
        # Assigning a Call to a Name (line 303):
        
        # Assigning a Call to a Name (line 303):
        
        # Call to array(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Obtaining an instance of the builtin type 'list' (line 303)
        list_315626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 303)
        # Adding element type (line 303)
        float_315627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, float_315627)
        # Adding element type (line 303)
        float_315628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, float_315628)
        # Adding element type (line 303)
        float_315629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 44), 'float')
        # Getting the type of 'width' (line 303)
        width_315630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 'width', False)
        # Applying the binary operator '-' (line 303)
        result_sub_315631 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 44), '-', float_315629, width_315630)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, result_sub_315631)
        # Adding element type (line 303)
        float_315632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 55), 'float')
        # Getting the type of 'width' (line 303)
        width_315633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 59), 'width', False)
        # Applying the binary operator '+' (line 303)
        result_add_315634 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 55), '+', float_315632, width_315633)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, result_add_315634)
        # Adding element type (line 303)
        float_315635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, float_315635)
        # Adding element type (line 303)
        float_315636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 32), list_315626, float_315636)
        
        # Processing the call keyword arguments (line 303)
        kwargs_315637 = {}
        # Getting the type of 'np' (line 303)
        np_315624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 303)
        array_315625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 23), np_315624, 'array')
        # Calling array(args, kwargs) (line 303)
        array_call_result_315638 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), array_315625, *[list_315626], **kwargs_315637)
        
        # Assigning a type to the variable 'freq_samples' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'freq_samples', array_call_result_315638)
        
        # Assigning a Call to a Tuple (line 304):
        
        # Assigning a Subscript to a Name (line 304):
        
        # Obtaining the type of the subscript
        int_315639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 8), 'int')
        
        # Call to freqz(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'taps' (line 304)
        taps_315641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'taps', False)
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'np' (line 304)
        np_315642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 304)
        pi_315643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 43), np_315642, 'pi')
        # Getting the type of 'freq_samples' (line 304)
        freq_samples_315644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 304)
        result_mul_315645 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 43), '*', pi_315643, freq_samples_315644)
        
        keyword_315646 = result_mul_315645
        kwargs_315647 = {'worN': keyword_315646}
        # Getting the type of 'freqz' (line 304)
        freqz_315640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 304)
        freqz_call_result_315648 = invoke(stypy.reporting.localization.Localization(__file__, 304, 26), freqz_315640, *[taps_315641], **kwargs_315647)
        
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___315649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), freqz_call_result_315648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_315650 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), getitem___315649, int_315639)
        
        # Assigning a type to the variable 'tuple_var_assignment_313931' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_313931', subscript_call_result_315650)
        
        # Assigning a Subscript to a Name (line 304):
        
        # Obtaining the type of the subscript
        int_315651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 8), 'int')
        
        # Call to freqz(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'taps' (line 304)
        taps_315653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'taps', False)
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'np' (line 304)
        np_315654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 304)
        pi_315655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 43), np_315654, 'pi')
        # Getting the type of 'freq_samples' (line 304)
        freq_samples_315656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 304)
        result_mul_315657 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 43), '*', pi_315655, freq_samples_315656)
        
        keyword_315658 = result_mul_315657
        kwargs_315659 = {'worN': keyword_315658}
        # Getting the type of 'freqz' (line 304)
        freqz_315652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 304)
        freqz_call_result_315660 = invoke(stypy.reporting.localization.Localization(__file__, 304, 26), freqz_315652, *[taps_315653], **kwargs_315659)
        
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___315661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), freqz_call_result_315660, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_315662 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), getitem___315661, int_315651)
        
        # Assigning a type to the variable 'tuple_var_assignment_313932' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_313932', subscript_call_result_315662)
        
        # Assigning a Name to a Name (line 304):
        # Getting the type of 'tuple_var_assignment_313931' (line 304)
        tuple_var_assignment_313931_315663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_313931')
        # Assigning a type to the variable 'freqs' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'freqs', tuple_var_assignment_313931_315663)
        
        # Assigning a Name to a Name (line 304):
        # Getting the type of 'tuple_var_assignment_313932' (line 304)
        tuple_var_assignment_313932_315664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'tuple_var_assignment_313932')
        # Assigning a type to the variable 'response' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'response', tuple_var_assignment_313932_315664)
        
        # Call to assert_array_almost_equal(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Call to abs(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'response' (line 305)
        response_315668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 41), 'response', False)
        # Processing the call keyword arguments (line 305)
        kwargs_315669 = {}
        # Getting the type of 'np' (line 305)
        np_315666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 305)
        abs_315667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 34), np_315666, 'abs')
        # Calling abs(args, kwargs) (line 305)
        abs_call_result_315670 = invoke(stypy.reporting.localization.Localization(__file__, 305, 34), abs_315667, *[response_315668], **kwargs_315669)
        
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_315671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        float_315672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315672)
        # Adding element type (line 306)
        float_315673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315673)
        # Adding element type (line 306)
        float_315674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315674)
        # Adding element type (line 306)
        float_315675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315675)
        # Adding element type (line 306)
        float_315676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315676)
        # Adding element type (line 306)
        float_315677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 32), list_315671, float_315677)
        
        # Processing the call keyword arguments (line 305)
        int_315678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 72), 'int')
        keyword_315679 = int_315678
        kwargs_315680 = {'decimal': keyword_315679}
        # Getting the type of 'assert_array_almost_equal' (line 305)
        assert_array_almost_equal_315665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 305)
        assert_array_almost_equal_call_result_315681 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), assert_array_almost_equal_315665, *[abs_call_result_315670, list_315671], **kwargs_315680)
        
        
        # ################# End of 'test02(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test02' in the type store
        # Getting the type of 'stypy_return_type' (line 294)
        stypy_return_type_315682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test02'
        return stypy_return_type_315682


    @norecursion
    def test03(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test03'
        module_type_store = module_type_store.open_function_context('test03', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test03.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test03.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test03.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test03.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test03')
        TestFirwin2.test03.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test03.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test03.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test03.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test03.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test03.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test03.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test03', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test03', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test03(...)' code ##################

        
        # Assigning a Num to a Name (line 309):
        
        # Assigning a Num to a Name (line 309):
        float_315683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 16), 'float')
        # Assigning a type to the variable 'width' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'width', float_315683)
        
        # Assigning a Call to a Tuple (line 310):
        
        # Assigning a Subscript to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_315684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        
        # Call to kaiserord(...): (line 310)
        # Processing the call arguments (line 310)
        int_315686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 32), 'int')
        # Getting the type of 'width' (line 310)
        width_315687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'width', False)
        # Processing the call keyword arguments (line 310)
        kwargs_315688 = {}
        # Getting the type of 'kaiserord' (line 310)
        kaiserord_315685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 310)
        kaiserord_call_result_315689 = invoke(stypy.reporting.localization.Localization(__file__, 310, 22), kaiserord_315685, *[int_315686, width_315687], **kwargs_315688)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___315690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), kaiserord_call_result_315689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_315691 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), getitem___315690, int_315684)
        
        # Assigning a type to the variable 'tuple_var_assignment_313933' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_313933', subscript_call_result_315691)
        
        # Assigning a Subscript to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_315692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 8), 'int')
        
        # Call to kaiserord(...): (line 310)
        # Processing the call arguments (line 310)
        int_315694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 32), 'int')
        # Getting the type of 'width' (line 310)
        width_315695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 37), 'width', False)
        # Processing the call keyword arguments (line 310)
        kwargs_315696 = {}
        # Getting the type of 'kaiserord' (line 310)
        kaiserord_315693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 22), 'kaiserord', False)
        # Calling kaiserord(args, kwargs) (line 310)
        kaiserord_call_result_315697 = invoke(stypy.reporting.localization.Localization(__file__, 310, 22), kaiserord_315693, *[int_315694, width_315695], **kwargs_315696)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___315698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), kaiserord_call_result_315697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_315699 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), getitem___315698, int_315692)
        
        # Assigning a type to the variable 'tuple_var_assignment_313934' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_313934', subscript_call_result_315699)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'tuple_var_assignment_313933' (line 310)
        tuple_var_assignment_313933_315700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_313933')
        # Assigning a type to the variable 'ntaps' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'ntaps', tuple_var_assignment_313933_315700)
        
        # Assigning a Name to a Name (line 310):
        # Getting the type of 'tuple_var_assignment_313934' (line 310)
        tuple_var_assignment_313934_315701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'tuple_var_assignment_313934')
        # Assigning a type to the variable 'beta' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'beta', tuple_var_assignment_313934_315701)
        
        # Assigning a BinOp to a Name (line 312):
        
        # Assigning a BinOp to a Name (line 312):
        
        # Call to int(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'ntaps' (line 312)
        ntaps_315703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'ntaps', False)
        # Processing the call keyword arguments (line 312)
        kwargs_315704 = {}
        # Getting the type of 'int' (line 312)
        int_315702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'int', False)
        # Calling int(args, kwargs) (line 312)
        int_call_result_315705 = invoke(stypy.reporting.localization.Localization(__file__, 312, 16), int_315702, *[ntaps_315703], **kwargs_315704)
        
        int_315706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 29), 'int')
        # Applying the binary operator '|' (line 312)
        result_or__315707 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 16), '|', int_call_result_315705, int_315706)
        
        # Assigning a type to the variable 'ntaps' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'ntaps', result_or__315707)
        
        # Assigning a List to a Name (line 313):
        
        # Assigning a List to a Name (line 313):
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_315708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        float_315709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315709)
        # Adding element type (line 313)
        float_315710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315710)
        # Adding element type (line 313)
        float_315711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315711)
        # Adding element type (line 313)
        float_315712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315712)
        # Adding element type (line 313)
        float_315713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315713)
        # Adding element type (line 313)
        float_315714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), list_315708, float_315714)
        
        # Assigning a type to the variable 'freq' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'freq', list_315708)
        
        # Assigning a List to a Name (line 314):
        
        # Assigning a List to a Name (line 314):
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_315715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        float_315716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315716)
        # Adding element type (line 314)
        float_315717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315717)
        # Adding element type (line 314)
        float_315718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315718)
        # Adding element type (line 314)
        float_315719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315719)
        # Adding element type (line 314)
        float_315720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315720)
        # Adding element type (line 314)
        float_315721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 15), list_315715, float_315721)
        
        # Assigning a type to the variable 'gain' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'gain', list_315715)
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to firwin2(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'ntaps' (line 315)
        ntaps_315723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 315)
        freq_315724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'freq', False)
        # Getting the type of 'gain' (line 315)
        gain_315725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 36), 'gain', False)
        # Processing the call keyword arguments (line 315)
        
        # Obtaining an instance of the builtin type 'tuple' (line 315)
        tuple_315726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 315)
        # Adding element type (line 315)
        str_315727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 50), 'str', 'kaiser')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 50), tuple_315726, str_315727)
        # Adding element type (line 315)
        # Getting the type of 'beta' (line 315)
        beta_315728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 60), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 50), tuple_315726, beta_315728)
        
        keyword_315729 = tuple_315726
        kwargs_315730 = {'window': keyword_315729}
        # Getting the type of 'firwin2' (line 315)
        firwin2_315722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 315)
        firwin2_call_result_315731 = invoke(stypy.reporting.localization.Localization(__file__, 315, 15), firwin2_315722, *[ntaps_315723, freq_315724, gain_315725], **kwargs_315730)
        
        # Assigning a type to the variable 'taps' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'taps', firwin2_call_result_315731)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to array(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_315734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        float_315735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, float_315735)
        # Adding element type (line 316)
        float_315736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 38), 'float')
        # Getting the type of 'width' (line 316)
        width_315737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 42), 'width', False)
        # Applying the binary operator '-' (line 316)
        result_sub_315738 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 38), '-', float_315736, width_315737)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, result_sub_315738)
        # Adding element type (line 316)
        float_315739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 49), 'float')
        # Getting the type of 'width' (line 316)
        width_315740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 53), 'width', False)
        # Applying the binary operator '+' (line 316)
        result_add_315741 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 49), '+', float_315739, width_315740)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, result_add_315741)
        # Adding element type (line 316)
        float_315742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, float_315742)
        # Adding element type (line 316)
        float_315743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 36), 'float')
        # Getting the type of 'width' (line 317)
        width_315744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 40), 'width', False)
        # Applying the binary operator '-' (line 317)
        result_sub_315745 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 36), '-', float_315743, width_315744)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, result_sub_315745)
        # Adding element type (line 316)
        float_315746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 47), 'float')
        # Getting the type of 'width' (line 317)
        width_315747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 51), 'width', False)
        # Applying the binary operator '+' (line 317)
        result_add_315748 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 47), '+', float_315746, width_315747)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, result_add_315748)
        # Adding element type (line 316)
        float_315749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, float_315749)
        # Adding element type (line 316)
        float_315750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 32), list_315734, float_315750)
        
        # Processing the call keyword arguments (line 316)
        kwargs_315751 = {}
        # Getting the type of 'np' (line 316)
        np_315732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 316)
        array_315733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), np_315732, 'array')
        # Calling array(args, kwargs) (line 316)
        array_call_result_315752 = invoke(stypy.reporting.localization.Localization(__file__, 316, 23), array_315733, *[list_315734], **kwargs_315751)
        
        # Assigning a type to the variable 'freq_samples' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'freq_samples', array_call_result_315752)
        
        # Assigning a Call to a Tuple (line 318):
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_315753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'int')
        
        # Call to freqz(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'taps' (line 318)
        taps_315755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'taps', False)
        # Processing the call keyword arguments (line 318)
        # Getting the type of 'np' (line 318)
        np_315756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 318)
        pi_315757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 43), np_315756, 'pi')
        # Getting the type of 'freq_samples' (line 318)
        freq_samples_315758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 318)
        result_mul_315759 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 43), '*', pi_315757, freq_samples_315758)
        
        keyword_315760 = result_mul_315759
        kwargs_315761 = {'worN': keyword_315760}
        # Getting the type of 'freqz' (line 318)
        freqz_315754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 318)
        freqz_call_result_315762 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), freqz_315754, *[taps_315755], **kwargs_315761)
        
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___315763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), freqz_call_result_315762, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_315764 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), getitem___315763, int_315753)
        
        # Assigning a type to the variable 'tuple_var_assignment_313935' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_313935', subscript_call_result_315764)
        
        # Assigning a Subscript to a Name (line 318):
        
        # Obtaining the type of the subscript
        int_315765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 8), 'int')
        
        # Call to freqz(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'taps' (line 318)
        taps_315767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'taps', False)
        # Processing the call keyword arguments (line 318)
        # Getting the type of 'np' (line 318)
        np_315768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 43), 'np', False)
        # Obtaining the member 'pi' of a type (line 318)
        pi_315769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 43), np_315768, 'pi')
        # Getting the type of 'freq_samples' (line 318)
        freq_samples_315770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 49), 'freq_samples', False)
        # Applying the binary operator '*' (line 318)
        result_mul_315771 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 43), '*', pi_315769, freq_samples_315770)
        
        keyword_315772 = result_mul_315771
        kwargs_315773 = {'worN': keyword_315772}
        # Getting the type of 'freqz' (line 318)
        freqz_315766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 318)
        freqz_call_result_315774 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), freqz_315766, *[taps_315767], **kwargs_315773)
        
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___315775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 8), freqz_call_result_315774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_315776 = invoke(stypy.reporting.localization.Localization(__file__, 318, 8), getitem___315775, int_315765)
        
        # Assigning a type to the variable 'tuple_var_assignment_313936' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_313936', subscript_call_result_315776)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_var_assignment_313935' (line 318)
        tuple_var_assignment_313935_315777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_313935')
        # Assigning a type to the variable 'freqs' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'freqs', tuple_var_assignment_313935_315777)
        
        # Assigning a Name to a Name (line 318):
        # Getting the type of 'tuple_var_assignment_313936' (line 318)
        tuple_var_assignment_313936_315778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'tuple_var_assignment_313936')
        # Assigning a type to the variable 'response' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'response', tuple_var_assignment_313936_315778)
        
        # Call to assert_array_almost_equal(...): (line 319)
        # Processing the call arguments (line 319)
        
        # Call to abs(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'response' (line 319)
        response_315782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 41), 'response', False)
        # Processing the call keyword arguments (line 319)
        kwargs_315783 = {}
        # Getting the type of 'np' (line 319)
        np_315780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 34), 'np', False)
        # Obtaining the member 'abs' of a type (line 319)
        abs_315781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 34), np_315780, 'abs')
        # Calling abs(args, kwargs) (line 319)
        abs_call_result_315784 = invoke(stypy.reporting.localization.Localization(__file__, 319, 34), abs_315781, *[response_315782], **kwargs_315783)
        
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_315785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        # Adding element type (line 320)
        float_315786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315786)
        # Adding element type (line 320)
        float_315787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315787)
        # Adding element type (line 320)
        float_315788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315788)
        # Adding element type (line 320)
        float_315789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315789)
        # Adding element type (line 320)
        float_315790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315790)
        # Adding element type (line 320)
        float_315791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315791)
        # Adding element type (line 320)
        float_315792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315792)
        # Adding element type (line 320)
        float_315793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 20), list_315785, float_315793)
        
        # Processing the call keyword arguments (line 319)
        int_315794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 70), 'int')
        keyword_315795 = int_315794
        kwargs_315796 = {'decimal': keyword_315795}
        # Getting the type of 'assert_array_almost_equal' (line 319)
        assert_array_almost_equal_315779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 319)
        assert_array_almost_equal_call_result_315797 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), assert_array_almost_equal_315779, *[abs_call_result_315784, list_315785], **kwargs_315796)
        
        
        # ################# End of 'test03(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test03' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_315798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315798)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test03'
        return stypy_return_type_315798


    @norecursion
    def test04(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test04'
        module_type_store = module_type_store.open_function_context('test04', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test04.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test04.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test04.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test04.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test04')
        TestFirwin2.test04.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test04.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test04.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test04.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test04.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test04.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test04.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test04', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test04', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test04(...)' code ##################

        str_315799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 8), 'str', 'Test firwin2 when window=None.')
        
        # Assigning a Num to a Name (line 324):
        
        # Assigning a Num to a Name (line 324):
        int_315800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'int')
        # Assigning a type to the variable 'ntaps' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'ntaps', int_315800)
        
        # Assigning a List to a Name (line 326):
        
        # Assigning a List to a Name (line 326):
        
        # Obtaining an instance of the builtin type 'list' (line 326)
        list_315801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 326)
        # Adding element type (line 326)
        float_315802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 15), list_315801, float_315802)
        # Adding element type (line 326)
        float_315803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 15), list_315801, float_315803)
        # Adding element type (line 326)
        float_315804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 15), list_315801, float_315804)
        # Adding element type (line 326)
        float_315805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 15), list_315801, float_315805)
        
        # Assigning a type to the variable 'freq' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'freq', list_315801)
        
        # Assigning a List to a Name (line 327):
        
        # Assigning a List to a Name (line 327):
        
        # Obtaining an instance of the builtin type 'list' (line 327)
        list_315806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 327)
        # Adding element type (line 327)
        float_315807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_315806, float_315807)
        # Adding element type (line 327)
        float_315808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_315806, float_315808)
        # Adding element type (line 327)
        float_315809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_315806, float_315809)
        # Adding element type (line 327)
        float_315810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 15), list_315806, float_315810)
        
        # Assigning a type to the variable 'gain' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'gain', list_315806)
        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to firwin2(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'ntaps' (line 328)
        ntaps_315812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 328)
        freq_315813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 30), 'freq', False)
        # Getting the type of 'gain' (line 328)
        gain_315814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 36), 'gain', False)
        # Processing the call keyword arguments (line 328)
        # Getting the type of 'None' (line 328)
        None_315815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 49), 'None', False)
        keyword_315816 = None_315815
        int_315817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 62), 'int')
        keyword_315818 = int_315817
        kwargs_315819 = {'nfreqs': keyword_315818, 'window': keyword_315816}
        # Getting the type of 'firwin2' (line 328)
        firwin2_315811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 328)
        firwin2_call_result_315820 = invoke(stypy.reporting.localization.Localization(__file__, 328, 15), firwin2_315811, *[ntaps_315812, freq_315813, gain_315814], **kwargs_315819)
        
        # Assigning a type to the variable 'taps' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'taps', firwin2_call_result_315820)
        
        # Assigning a BinOp to a Name (line 329):
        
        # Assigning a BinOp to a Name (line 329):
        float_315821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 16), 'float')
        # Getting the type of 'ntaps' (line 329)
        ntaps_315822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 23), 'ntaps')
        int_315823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 31), 'int')
        # Applying the binary operator '-' (line 329)
        result_sub_315824 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 23), '-', ntaps_315822, int_315823)
        
        # Applying the binary operator '*' (line 329)
        result_mul_315825 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 16), '*', float_315821, result_sub_315824)
        
        # Assigning a type to the variable 'alpha' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'alpha', result_mul_315825)
        
        # Assigning a BinOp to a Name (line 330):
        
        # Assigning a BinOp to a Name (line 330):
        
        # Call to arange(...): (line 330)
        # Processing the call arguments (line 330)
        int_315828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 22), 'int')
        # Getting the type of 'ntaps' (line 330)
        ntaps_315829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 25), 'ntaps', False)
        # Processing the call keyword arguments (line 330)
        kwargs_315830 = {}
        # Getting the type of 'np' (line 330)
        np_315826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 330)
        arange_315827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), np_315826, 'arange')
        # Calling arange(args, kwargs) (line 330)
        arange_call_result_315831 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), arange_315827, *[int_315828, ntaps_315829], **kwargs_315830)
        
        # Getting the type of 'alpha' (line 330)
        alpha_315832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 34), 'alpha')
        # Applying the binary operator '-' (line 330)
        result_sub_315833 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 12), '-', arange_call_result_315831, alpha_315832)
        
        # Assigning a type to the variable 'm' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'm', result_sub_315833)
        
        # Assigning a BinOp to a Name (line 331):
        
        # Assigning a BinOp to a Name (line 331):
        float_315834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 12), 'float')
        
        # Call to sinc(...): (line 331)
        # Processing the call arguments (line 331)
        float_315836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'float')
        # Getting the type of 'm' (line 331)
        m_315837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'm', False)
        # Applying the binary operator '*' (line 331)
        result_mul_315838 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 23), '*', float_315836, m_315837)
        
        # Processing the call keyword arguments (line 331)
        kwargs_315839 = {}
        # Getting the type of 'sinc' (line 331)
        sinc_315835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 18), 'sinc', False)
        # Calling sinc(args, kwargs) (line 331)
        sinc_call_result_315840 = invoke(stypy.reporting.localization.Localization(__file__, 331, 18), sinc_315835, *[result_mul_315838], **kwargs_315839)
        
        # Applying the binary operator '*' (line 331)
        result_mul_315841 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 12), '*', float_315834, sinc_call_result_315840)
        
        # Assigning a type to the variable 'h' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'h', result_mul_315841)
        
        # Call to assert_array_almost_equal(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'h' (line 332)
        h_315843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'h', False)
        # Getting the type of 'taps' (line 332)
        taps_315844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'taps', False)
        # Processing the call keyword arguments (line 332)
        kwargs_315845 = {}
        # Getting the type of 'assert_array_almost_equal' (line 332)
        assert_array_almost_equal_315842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 332)
        assert_array_almost_equal_call_result_315846 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), assert_array_almost_equal_315842, *[h_315843, taps_315844], **kwargs_315845)
        
        
        # ################# End of 'test04(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test04' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_315847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test04'
        return stypy_return_type_315847


    @norecursion
    def test05(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test05'
        module_type_store = module_type_store.open_function_context('test05', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test05.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test05.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test05.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test05.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test05')
        TestFirwin2.test05.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test05.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test05.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test05.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test05.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test05.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test05.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test05', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test05', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test05(...)' code ##################

        str_315848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 8), 'str', 'Test firwin2 for calculating Type IV filters')
        
        # Assigning a Num to a Name (line 336):
        
        # Assigning a Num to a Name (line 336):
        int_315849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 16), 'int')
        # Assigning a type to the variable 'ntaps' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'ntaps', int_315849)
        
        # Assigning a List to a Name (line 338):
        
        # Assigning a List to a Name (line 338):
        
        # Obtaining an instance of the builtin type 'list' (line 338)
        list_315850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 338)
        # Adding element type (line 338)
        float_315851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 15), list_315850, float_315851)
        # Adding element type (line 338)
        float_315852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 15), list_315850, float_315852)
        
        # Assigning a type to the variable 'freq' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'freq', list_315850)
        
        # Assigning a List to a Name (line 339):
        
        # Assigning a List to a Name (line 339):
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_315853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        # Adding element type (line 339)
        float_315854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 15), list_315853, float_315854)
        # Adding element type (line 339)
        float_315855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 15), list_315853, float_315855)
        
        # Assigning a type to the variable 'gain' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'gain', list_315853)
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to firwin2(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'ntaps' (line 340)
        ntaps_315857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 340)
        freq_315858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'freq', False)
        # Getting the type of 'gain' (line 340)
        gain_315859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 36), 'gain', False)
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'None' (line 340)
        None_315860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 49), 'None', False)
        keyword_315861 = None_315860
        # Getting the type of 'True' (line 340)
        True_315862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 69), 'True', False)
        keyword_315863 = True_315862
        kwargs_315864 = {'window': keyword_315861, 'antisymmetric': keyword_315863}
        # Getting the type of 'firwin2' (line 340)
        firwin2_315856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 340)
        firwin2_call_result_315865 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), firwin2_315856, *[ntaps_315857, freq_315858, gain_315859], **kwargs_315864)
        
        # Assigning a type to the variable 'taps' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'taps', firwin2_call_result_315865)
        
        # Call to assert_array_almost_equal(...): (line 341)
        # Processing the call arguments (line 341)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 341)
        ntaps_315867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 41), 'ntaps', False)
        int_315868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 50), 'int')
        # Applying the binary operator '//' (line 341)
        result_floordiv_315869 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 41), '//', ntaps_315867, int_315868)
        
        slice_315870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 34), None, result_floordiv_315869, None)
        # Getting the type of 'taps' (line 341)
        taps_315871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___315872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 34), taps_315871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_315873 = invoke(stypy.reporting.localization.Localization(__file__, 341, 34), getitem___315872, slice_315870)
        
        
        
        # Obtaining the type of the subscript
        int_315874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 75), 'int')
        slice_315875 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 55), None, None, int_315874)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 341)
        ntaps_315876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 60), 'ntaps', False)
        int_315877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 69), 'int')
        # Applying the binary operator '//' (line 341)
        result_floordiv_315878 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 60), '//', ntaps_315876, int_315877)
        
        slice_315879 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 341, 55), result_floordiv_315878, None, None)
        # Getting the type of 'taps' (line 341)
        taps_315880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___315881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 55), taps_315880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_315882 = invoke(stypy.reporting.localization.Localization(__file__, 341, 55), getitem___315881, slice_315879)
        
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___315883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 55), subscript_call_result_315882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_315884 = invoke(stypy.reporting.localization.Localization(__file__, 341, 55), getitem___315883, slice_315875)
        
        # Applying the 'usub' unary operator (line 341)
        result___neg___315885 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 54), 'usub', subscript_call_result_315884)
        
        # Processing the call keyword arguments (line 341)
        kwargs_315886 = {}
        # Getting the type of 'assert_array_almost_equal' (line 341)
        assert_array_almost_equal_315866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 341)
        assert_array_almost_equal_call_result_315887 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), assert_array_almost_equal_315866, *[subscript_call_result_315873, result___neg___315885], **kwargs_315886)
        
        
        # Assigning a Call to a Tuple (line 343):
        
        # Assigning a Subscript to a Name (line 343):
        
        # Obtaining the type of the subscript
        int_315888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        
        # Call to freqz(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'taps' (line 343)
        taps_315890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'taps', False)
        # Processing the call keyword arguments (line 343)
        int_315891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 43), 'int')
        keyword_315892 = int_315891
        kwargs_315893 = {'worN': keyword_315892}
        # Getting the type of 'freqz' (line 343)
        freqz_315889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 343)
        freqz_call_result_315894 = invoke(stypy.reporting.localization.Localization(__file__, 343, 26), freqz_315889, *[taps_315890], **kwargs_315893)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___315895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), freqz_call_result_315894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_315896 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), getitem___315895, int_315888)
        
        # Assigning a type to the variable 'tuple_var_assignment_313937' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'tuple_var_assignment_313937', subscript_call_result_315896)
        
        # Assigning a Subscript to a Name (line 343):
        
        # Obtaining the type of the subscript
        int_315897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 8), 'int')
        
        # Call to freqz(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'taps' (line 343)
        taps_315899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 32), 'taps', False)
        # Processing the call keyword arguments (line 343)
        int_315900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 43), 'int')
        keyword_315901 = int_315900
        kwargs_315902 = {'worN': keyword_315901}
        # Getting the type of 'freqz' (line 343)
        freqz_315898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'freqz', False)
        # Calling freqz(args, kwargs) (line 343)
        freqz_call_result_315903 = invoke(stypy.reporting.localization.Localization(__file__, 343, 26), freqz_315898, *[taps_315899], **kwargs_315902)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___315904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), freqz_call_result_315903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_315905 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), getitem___315904, int_315897)
        
        # Assigning a type to the variable 'tuple_var_assignment_313938' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'tuple_var_assignment_313938', subscript_call_result_315905)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'tuple_var_assignment_313937' (line 343)
        tuple_var_assignment_313937_315906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'tuple_var_assignment_313937')
        # Assigning a type to the variable 'freqs' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'freqs', tuple_var_assignment_313937_315906)
        
        # Assigning a Name to a Name (line 343):
        # Getting the type of 'tuple_var_assignment_313938' (line 343)
        tuple_var_assignment_313938_315907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'tuple_var_assignment_313938')
        # Assigning a type to the variable 'response' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'response', tuple_var_assignment_313938_315907)
        
        # Call to assert_array_almost_equal(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Call to abs(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'response' (line 344)
        response_315910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 38), 'response', False)
        # Processing the call keyword arguments (line 344)
        kwargs_315911 = {}
        # Getting the type of 'abs' (line 344)
        abs_315909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 34), 'abs', False)
        # Calling abs(args, kwargs) (line 344)
        abs_call_result_315912 = invoke(stypy.reporting.localization.Localization(__file__, 344, 34), abs_315909, *[response_315910], **kwargs_315911)
        
        # Getting the type of 'freqs' (line 344)
        freqs_315913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 49), 'freqs', False)
        # Getting the type of 'np' (line 344)
        np_315914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 57), 'np', False)
        # Obtaining the member 'pi' of a type (line 344)
        pi_315915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 57), np_315914, 'pi')
        # Applying the binary operator 'div' (line 344)
        result_div_315916 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 49), 'div', freqs_315913, pi_315915)
        
        # Processing the call keyword arguments (line 344)
        int_315917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 72), 'int')
        keyword_315918 = int_315917
        kwargs_315919 = {'decimal': keyword_315918}
        # Getting the type of 'assert_array_almost_equal' (line 344)
        assert_array_almost_equal_315908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 344)
        assert_array_almost_equal_call_result_315920 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), assert_array_almost_equal_315908, *[abs_call_result_315912, result_div_315916], **kwargs_315919)
        
        
        # ################# End of 'test05(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test05' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_315921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_315921)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test05'
        return stypy_return_type_315921


    @norecursion
    def test06(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test06'
        module_type_store = module_type_store.open_function_context('test06', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test06.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test06.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test06.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test06.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test06')
        TestFirwin2.test06.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test06.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test06.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test06.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test06.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test06.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test06.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test06', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test06', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test06(...)' code ##################

        str_315922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 8), 'str', 'Test firwin2 for calculating Type III filters')
        
        # Assigning a Num to a Name (line 348):
        
        # Assigning a Num to a Name (line 348):
        int_315923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 16), 'int')
        # Assigning a type to the variable 'ntaps' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'ntaps', int_315923)
        
        # Assigning a List to a Name (line 350):
        
        # Assigning a List to a Name (line 350):
        
        # Obtaining an instance of the builtin type 'list' (line 350)
        list_315924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 350)
        # Adding element type (line 350)
        float_315925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 15), list_315924, float_315925)
        # Adding element type (line 350)
        float_315926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 15), list_315924, float_315926)
        # Adding element type (line 350)
        float_315927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 15), list_315924, float_315927)
        # Adding element type (line 350)
        float_315928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 15), list_315924, float_315928)
        
        # Assigning a type to the variable 'freq' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'freq', list_315924)
        
        # Assigning a List to a Name (line 351):
        
        # Assigning a List to a Name (line 351):
        
        # Obtaining an instance of the builtin type 'list' (line 351)
        list_315929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 351)
        # Adding element type (line 351)
        float_315930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 15), list_315929, float_315930)
        # Adding element type (line 351)
        float_315931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 21), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 15), list_315929, float_315931)
        # Adding element type (line 351)
        float_315932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 15), list_315929, float_315932)
        # Adding element type (line 351)
        float_315933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 15), list_315929, float_315933)
        
        # Assigning a type to the variable 'gain' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'gain', list_315929)
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to firwin2(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'ntaps' (line 352)
        ntaps_315935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 23), 'ntaps', False)
        # Getting the type of 'freq' (line 352)
        freq_315936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'freq', False)
        # Getting the type of 'gain' (line 352)
        gain_315937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 36), 'gain', False)
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'None' (line 352)
        None_315938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 49), 'None', False)
        keyword_315939 = None_315938
        # Getting the type of 'True' (line 352)
        True_315940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 69), 'True', False)
        keyword_315941 = True_315940
        kwargs_315942 = {'window': keyword_315939, 'antisymmetric': keyword_315941}
        # Getting the type of 'firwin2' (line 352)
        firwin2_315934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 352)
        firwin2_call_result_315943 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), firwin2_315934, *[ntaps_315935, freq_315936, gain_315937], **kwargs_315942)
        
        # Assigning a type to the variable 'taps' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'taps', firwin2_call_result_315943)
        
        # Call to assert_equal(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 353)
        ntaps_315945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 26), 'ntaps', False)
        int_315946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 35), 'int')
        # Applying the binary operator '//' (line 353)
        result_floordiv_315947 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 26), '//', ntaps_315945, int_315946)
        
        # Getting the type of 'taps' (line 353)
        taps_315948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 353)
        getitem___315949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 21), taps_315948, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 353)
        subscript_call_result_315950 = invoke(stypy.reporting.localization.Localization(__file__, 353, 21), getitem___315949, result_floordiv_315947)
        
        float_315951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 39), 'float')
        # Processing the call keyword arguments (line 353)
        kwargs_315952 = {}
        # Getting the type of 'assert_equal' (line 353)
        assert_equal_315944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 353)
        assert_equal_call_result_315953 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert_equal_315944, *[subscript_call_result_315950, float_315951], **kwargs_315952)
        
        
        # Call to assert_array_almost_equal(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 354)
        ntaps_315955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'ntaps', False)
        int_315956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 50), 'int')
        # Applying the binary operator '//' (line 354)
        result_floordiv_315957 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 41), '//', ntaps_315955, int_315956)
        
        slice_315958 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 34), None, result_floordiv_315957, None)
        # Getting the type of 'taps' (line 354)
        taps_315959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___315960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 34), taps_315959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_315961 = invoke(stypy.reporting.localization.Localization(__file__, 354, 34), getitem___315960, slice_315958)
        
        
        
        # Obtaining the type of the subscript
        int_315962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 79), 'int')
        slice_315963 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 55), None, None, int_315962)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ntaps' (line 354)
        ntaps_315964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 60), 'ntaps', False)
        int_315965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 69), 'int')
        # Applying the binary operator '//' (line 354)
        result_floordiv_315966 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 60), '//', ntaps_315964, int_315965)
        
        int_315967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 73), 'int')
        # Applying the binary operator '+' (line 354)
        result_add_315968 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 60), '+', result_floordiv_315966, int_315967)
        
        slice_315969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 354, 55), result_add_315968, None, None)
        # Getting the type of 'taps' (line 354)
        taps_315970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 55), 'taps', False)
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___315971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 55), taps_315970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_315972 = invoke(stypy.reporting.localization.Localization(__file__, 354, 55), getitem___315971, slice_315969)
        
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___315973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 55), subscript_call_result_315972, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_315974 = invoke(stypy.reporting.localization.Localization(__file__, 354, 55), getitem___315973, slice_315963)
        
        # Applying the 'usub' unary operator (line 354)
        result___neg___315975 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 54), 'usub', subscript_call_result_315974)
        
        # Processing the call keyword arguments (line 354)
        kwargs_315976 = {}
        # Getting the type of 'assert_array_almost_equal' (line 354)
        assert_array_almost_equal_315954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 354)
        assert_array_almost_equal_call_result_315977 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), assert_array_almost_equal_315954, *[subscript_call_result_315961, result___neg___315975], **kwargs_315976)
        
        
        # Assigning a Call to a Tuple (line 356):
        
        # Assigning a Subscript to a Name (line 356):
        
        # Obtaining the type of the subscript
        int_315978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 8), 'int')
        
        # Call to freqz(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'taps' (line 356)
        taps_315980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'taps', False)
        # Processing the call keyword arguments (line 356)
        int_315981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 44), 'int')
        keyword_315982 = int_315981
        kwargs_315983 = {'worN': keyword_315982}
        # Getting the type of 'freqz' (line 356)
        freqz_315979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'freqz', False)
        # Calling freqz(args, kwargs) (line 356)
        freqz_call_result_315984 = invoke(stypy.reporting.localization.Localization(__file__, 356, 27), freqz_315979, *[taps_315980], **kwargs_315983)
        
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___315985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), freqz_call_result_315984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_315986 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), getitem___315985, int_315978)
        
        # Assigning a type to the variable 'tuple_var_assignment_313939' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'tuple_var_assignment_313939', subscript_call_result_315986)
        
        # Assigning a Subscript to a Name (line 356):
        
        # Obtaining the type of the subscript
        int_315987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 8), 'int')
        
        # Call to freqz(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'taps' (line 356)
        taps_315989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'taps', False)
        # Processing the call keyword arguments (line 356)
        int_315990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 44), 'int')
        keyword_315991 = int_315990
        kwargs_315992 = {'worN': keyword_315991}
        # Getting the type of 'freqz' (line 356)
        freqz_315988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'freqz', False)
        # Calling freqz(args, kwargs) (line 356)
        freqz_call_result_315993 = invoke(stypy.reporting.localization.Localization(__file__, 356, 27), freqz_315988, *[taps_315989], **kwargs_315992)
        
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___315994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), freqz_call_result_315993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_315995 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), getitem___315994, int_315987)
        
        # Assigning a type to the variable 'tuple_var_assignment_313940' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'tuple_var_assignment_313940', subscript_call_result_315995)
        
        # Assigning a Name to a Name (line 356):
        # Getting the type of 'tuple_var_assignment_313939' (line 356)
        tuple_var_assignment_313939_315996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'tuple_var_assignment_313939')
        # Assigning a type to the variable 'freqs' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'freqs', tuple_var_assignment_313939_315996)
        
        # Assigning a Name to a Name (line 356):
        # Getting the type of 'tuple_var_assignment_313940' (line 356)
        tuple_var_assignment_313940_315997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'tuple_var_assignment_313940')
        # Assigning a type to the variable 'response1' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 15), 'response1', tuple_var_assignment_313940_315997)
        
        # Assigning a Call to a Name (line 357):
        
        # Assigning a Call to a Name (line 357):
        
        # Call to interp(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'freqs' (line 357)
        freqs_316000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 30), 'freqs', False)
        # Getting the type of 'np' (line 357)
        np_316001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 38), 'np', False)
        # Obtaining the member 'pi' of a type (line 357)
        pi_316002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 38), np_316001, 'pi')
        # Applying the binary operator 'div' (line 357)
        result_div_316003 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 30), 'div', freqs_316000, pi_316002)
        
        # Getting the type of 'freq' (line 357)
        freq_316004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 45), 'freq', False)
        # Getting the type of 'gain' (line 357)
        gain_316005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 51), 'gain', False)
        # Processing the call keyword arguments (line 357)
        kwargs_316006 = {}
        # Getting the type of 'np' (line 357)
        np_315998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'np', False)
        # Obtaining the member 'interp' of a type (line 357)
        interp_315999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), np_315998, 'interp')
        # Calling interp(args, kwargs) (line 357)
        interp_call_result_316007 = invoke(stypy.reporting.localization.Localization(__file__, 357, 20), interp_315999, *[result_div_316003, freq_316004, gain_316005], **kwargs_316006)
        
        # Assigning a type to the variable 'response2' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'response2', interp_call_result_316007)
        
        # Call to assert_array_almost_equal(...): (line 358)
        # Processing the call arguments (line 358)
        
        # Call to abs(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'response1' (line 358)
        response1_316010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 38), 'response1', False)
        # Processing the call keyword arguments (line 358)
        kwargs_316011 = {}
        # Getting the type of 'abs' (line 358)
        abs_316009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'abs', False)
        # Calling abs(args, kwargs) (line 358)
        abs_call_result_316012 = invoke(stypy.reporting.localization.Localization(__file__, 358, 34), abs_316009, *[response1_316010], **kwargs_316011)
        
        # Getting the type of 'response2' (line 358)
        response2_316013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 50), 'response2', False)
        # Processing the call keyword arguments (line 358)
        int_316014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 69), 'int')
        keyword_316015 = int_316014
        kwargs_316016 = {'decimal': keyword_316015}
        # Getting the type of 'assert_array_almost_equal' (line 358)
        assert_array_almost_equal_316008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 358)
        assert_array_almost_equal_call_result_316017 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), assert_array_almost_equal_316008, *[abs_call_result_316012, response2_316013], **kwargs_316016)
        
        
        # ################# End of 'test06(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test06' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_316018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test06'
        return stypy_return_type_316018


    @norecursion
    def test_fs_nyq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fs_nyq'
        module_type_store = module_type_store.open_function_context('test_fs_nyq', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_localization', localization)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_function_name', 'TestFirwin2.test_fs_nyq')
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirwin2.test_fs_nyq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.test_fs_nyq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fs_nyq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fs_nyq(...)' code ##################

        
        # Assigning a Call to a Name (line 361):
        
        # Assigning a Call to a Name (line 361):
        
        # Call to firwin2(...): (line 361)
        # Processing the call arguments (line 361)
        int_316020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 24), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_316021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        float_316022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 28), list_316021, float_316022)
        # Adding element type (line 361)
        float_316023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 28), list_316021, float_316023)
        # Adding element type (line 361)
        float_316024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 28), list_316021, float_316024)
        
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_316025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        float_316026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 45), list_316025, float_316026)
        # Adding element type (line 361)
        float_316027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 45), list_316025, float_316027)
        # Adding element type (line 361)
        float_316028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 45), list_316025, float_316028)
        
        # Processing the call keyword arguments (line 361)
        kwargs_316029 = {}
        # Getting the type of 'firwin2' (line 361)
        firwin2_316019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 361)
        firwin2_call_result_316030 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), firwin2_316019, *[int_316020, list_316021, list_316025], **kwargs_316029)
        
        # Assigning a type to the variable 'taps1' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'taps1', firwin2_call_result_316030)
        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to firwin2(...): (line 362)
        # Processing the call arguments (line 362)
        int_316032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 362)
        list_316033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 362)
        # Adding element type (line 362)
        float_316034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 28), list_316033, float_316034)
        # Adding element type (line 362)
        float_316035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 28), list_316033, float_316035)
        # Adding element type (line 362)
        float_316036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 28), list_316033, float_316036)
        
        
        # Obtaining an instance of the builtin type 'list' (line 362)
        list_316037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 362)
        # Adding element type (line 362)
        float_316038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 47), list_316037, float_316038)
        # Adding element type (line 362)
        float_316039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 47), list_316037, float_316039)
        # Adding element type (line 362)
        float_316040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 47), list_316037, float_316040)
        
        # Processing the call keyword arguments (line 362)
        float_316041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 67), 'float')
        keyword_316042 = float_316041
        kwargs_316043 = {'fs': keyword_316042}
        # Getting the type of 'firwin2' (line 362)
        firwin2_316031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 362)
        firwin2_call_result_316044 = invoke(stypy.reporting.localization.Localization(__file__, 362, 16), firwin2_316031, *[int_316032, list_316033, list_316037], **kwargs_316043)
        
        # Assigning a type to the variable 'taps2' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'taps2', firwin2_call_result_316044)
        
        # Call to assert_array_almost_equal(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'taps1' (line 363)
        taps1_316046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 34), 'taps1', False)
        # Getting the type of 'taps2' (line 363)
        taps2_316047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 41), 'taps2', False)
        # Processing the call keyword arguments (line 363)
        kwargs_316048 = {}
        # Getting the type of 'assert_array_almost_equal' (line 363)
        assert_array_almost_equal_316045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 363)
        assert_array_almost_equal_call_result_316049 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), assert_array_almost_equal_316045, *[taps1_316046, taps2_316047], **kwargs_316048)
        
        
        # Assigning a Call to a Name (line 364):
        
        # Assigning a Call to a Name (line 364):
        
        # Call to firwin2(...): (line 364)
        # Processing the call arguments (line 364)
        int_316051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 24), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_316052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        float_316053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), list_316052, float_316053)
        # Adding element type (line 364)
        float_316054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), list_316052, float_316054)
        # Adding element type (line 364)
        float_316055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 28), list_316052, float_316055)
        
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_316056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        float_316057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 47), list_316056, float_316057)
        # Adding element type (line 364)
        float_316058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 47), list_316056, float_316058)
        # Adding element type (line 364)
        float_316059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 47), list_316056, float_316059)
        
        # Processing the call keyword arguments (line 364)
        float_316060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 68), 'float')
        keyword_316061 = float_316060
        kwargs_316062 = {'nyq': keyword_316061}
        # Getting the type of 'firwin2' (line 364)
        firwin2_316050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 16), 'firwin2', False)
        # Calling firwin2(args, kwargs) (line 364)
        firwin2_call_result_316063 = invoke(stypy.reporting.localization.Localization(__file__, 364, 16), firwin2_316050, *[int_316051, list_316052, list_316056], **kwargs_316062)
        
        # Assigning a type to the variable 'taps2' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'taps2', firwin2_call_result_316063)
        
        # Call to assert_array_almost_equal(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'taps1' (line 365)
        taps1_316065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'taps1', False)
        # Getting the type of 'taps2' (line 365)
        taps2_316066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 41), 'taps2', False)
        # Processing the call keyword arguments (line 365)
        kwargs_316067 = {}
        # Getting the type of 'assert_array_almost_equal' (line 365)
        assert_array_almost_equal_316064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 365)
        assert_array_almost_equal_call_result_316068 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), assert_array_almost_equal_316064, *[taps1_316065, taps2_316066], **kwargs_316067)
        
        
        # ################# End of 'test_fs_nyq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fs_nyq' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_316069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fs_nyq'
        return stypy_return_type_316069


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 249, 0, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirwin2.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFirwin2' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'TestFirwin2', TestFirwin2)
# Declaration of the 'TestRemez' class

class TestRemez(object, ):

    @norecursion
    def test_bad_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_args'
        module_type_store = module_type_store.open_function_context('test_bad_args', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_localization', localization)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_function_name', 'TestRemez.test_bad_args')
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRemez.test_bad_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRemez.test_bad_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_args(...)' code ##################

        
        # Call to assert_raises(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'ValueError' (line 370)
        ValueError_316071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 22), 'ValueError', False)
        # Getting the type of 'remez' (line 370)
        remez_316072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 34), 'remez', False)
        int_316073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 370)
        list_316074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 370)
        # Adding element type (line 370)
        float_316075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 45), list_316074, float_316075)
        # Adding element type (line 370)
        float_316076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 45), list_316074, float_316076)
        
        
        # Obtaining an instance of the builtin type 'list' (line 370)
        list_316077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 370)
        # Adding element type (line 370)
        int_316078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 57), list_316077, int_316078)
        
        # Processing the call keyword arguments (line 370)
        str_316079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 67), 'str', 'pooka')
        keyword_316080 = str_316079
        kwargs_316081 = {'type': keyword_316080}
        # Getting the type of 'assert_raises' (line 370)
        assert_raises_316070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 370)
        assert_raises_call_result_316082 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert_raises_316070, *[ValueError_316071, remez_316072, int_316073, list_316074, list_316077], **kwargs_316081)
        
        
        # ################# End of 'test_bad_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_args' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_316083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_args'
        return stypy_return_type_316083


    @norecursion
    def test_hilbert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hilbert'
        module_type_store = module_type_store.open_function_context('test_hilbert', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_localization', localization)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_function_name', 'TestRemez.test_hilbert')
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_param_names_list', [])
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRemez.test_hilbert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRemez.test_hilbert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hilbert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hilbert(...)' code ##################

        
        # Assigning a Num to a Name (line 373):
        
        # Assigning a Num to a Name (line 373):
        int_316084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 12), 'int')
        # Assigning a type to the variable 'N' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'N', int_316084)
        
        # Assigning a Num to a Name (line 374):
        
        # Assigning a Num to a Name (line 374):
        float_316085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 12), 'float')
        # Assigning a type to the variable 'a' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'a', float_316085)
        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to remez(...): (line 377)
        # Processing the call arguments (line 377)
        int_316087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_316088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'a' (line 377)
        a_316089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 22), list_316088, a_316089)
        # Adding element type (line 377)
        float_316090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 26), 'float')
        # Getting the type of 'a' (line 377)
        a_316091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'a', False)
        # Applying the binary operator '-' (line 377)
        result_sub_316092 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 26), '-', float_316090, a_316091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 22), list_316088, result_sub_316092)
        
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_316093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        int_316094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 34), list_316093, int_316094)
        
        # Processing the call keyword arguments (line 377)
        str_316095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 44), 'str', 'hilbert')
        keyword_316096 = str_316095
        kwargs_316097 = {'type': keyword_316096}
        # Getting the type of 'remez' (line 377)
        remez_316086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'remez', False)
        # Calling remez(args, kwargs) (line 377)
        remez_call_result_316098 = invoke(stypy.reporting.localization.Localization(__file__, 377, 12), remez_316086, *[int_316087, list_316088, list_316093], **kwargs_316097)
        
        # Assigning a type to the variable 'h' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'h', remez_call_result_316098)
        
        # Call to assert_(...): (line 380)
        # Processing the call arguments (line 380)
        
        
        # Call to len(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'h' (line 380)
        h_316101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'h', False)
        # Processing the call keyword arguments (line 380)
        kwargs_316102 = {}
        # Getting the type of 'len' (line 380)
        len_316100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'len', False)
        # Calling len(args, kwargs) (line 380)
        len_call_result_316103 = invoke(stypy.reporting.localization.Localization(__file__, 380, 16), len_316100, *[h_316101], **kwargs_316102)
        
        # Getting the type of 'N' (line 380)
        N_316104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'N', False)
        # Applying the binary operator '==' (line 380)
        result_eq_316105 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 16), '==', len_call_result_316103, N_316104)
        
        str_316106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 29), 'str', 'Number of Taps')
        # Processing the call keyword arguments (line 380)
        kwargs_316107 = {}
        # Getting the type of 'assert_' (line 380)
        assert__316099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 380)
        assert__call_result_316108 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), assert__316099, *[result_eq_316105, str_316106], **kwargs_316107)
        
        
        # Call to assert_array_almost_equal(...): (line 383)
        # Processing the call arguments (line 383)
        
        # Obtaining the type of the subscript
        # Getting the type of 'N' (line 383)
        N_316110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 38), 'N', False)
        int_316111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 40), 'int')
        # Applying the binary operator '-' (line 383)
        result_sub_316112 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 38), '-', N_316110, int_316111)
        
        int_316113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 44), 'int')
        # Applying the binary operator '//' (line 383)
        result_floordiv_316114 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 37), '//', result_sub_316112, int_316113)
        
        slice_316115 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 34), None, result_floordiv_316114, None)
        # Getting the type of 'h' (line 383)
        h_316116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 34), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___316117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 34), h_316116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 383)
        subscript_call_result_316118 = invoke(stypy.reporting.localization.Localization(__file__, 383, 34), getitem___316117, slice_316115)
        
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'N' (line 383)
        N_316119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 54), 'N', False)
        int_316120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 56), 'int')
        # Applying the binary operator '-' (line 383)
        result_sub_316121 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 54), '-', N_316119, int_316120)
        
        # Applying the 'usub' unary operator (line 383)
        result___neg___316122 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 52), 'usub', result_sub_316121)
        
        int_316123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 60), 'int')
        # Applying the binary operator '//' (line 383)
        result_floordiv_316124 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 52), '//', result___neg___316122, int_316123)
        
        int_316125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 62), 'int')
        # Applying the binary operator '-' (line 383)
        result_sub_316126 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 52), '-', result_floordiv_316124, int_316125)
        
        int_316127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 64), 'int')
        slice_316128 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 49), None, result_sub_316126, int_316127)
        # Getting the type of 'h' (line 383)
        h_316129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 49), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___316130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 49), h_316129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 383)
        subscript_call_result_316131 = invoke(stypy.reporting.localization.Localization(__file__, 383, 49), getitem___316130, slice_316128)
        
        # Applying the 'usub' unary operator (line 383)
        result___neg___316132 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 48), 'usub', subscript_call_result_316131)
        
        # Processing the call keyword arguments (line 383)
        kwargs_316133 = {}
        # Getting the type of 'assert_array_almost_equal' (line 383)
        assert_array_almost_equal_316109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 383)
        assert_array_almost_equal_call_result_316134 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), assert_array_almost_equal_316109, *[subscript_call_result_316118, result___neg___316132], **kwargs_316133)
        
        
        # Call to assert_(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Call to all(...): (line 387)
        # Processing the call keyword arguments (line 387)
        kwargs_316148 = {}
        
        
        # Call to abs(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Obtaining the type of the subscript
        int_316137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 23), 'int')
        int_316138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 26), 'int')
        slice_316139 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 387, 21), int_316137, None, int_316138)
        # Getting the type of 'h' (line 387)
        h_316140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 21), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___316141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 21), h_316140, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_316142 = invoke(stypy.reporting.localization.Localization(__file__, 387, 21), getitem___316141, slice_316139)
        
        # Processing the call keyword arguments (line 387)
        kwargs_316143 = {}
        # Getting the type of 'abs' (line 387)
        abs_316136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'abs', False)
        # Calling abs(args, kwargs) (line 387)
        abs_call_result_316144 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), abs_316136, *[subscript_call_result_316142], **kwargs_316143)
        
        float_316145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 32), 'float')
        # Applying the binary operator '<' (line 387)
        result_lt_316146 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 17), '<', abs_call_result_316144, float_316145)
        
        # Obtaining the member 'all' of a type (line 387)
        all_316147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 17), result_lt_316146, 'all')
        # Calling all(args, kwargs) (line 387)
        all_call_result_316149 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), all_316147, *[], **kwargs_316148)
        
        str_316150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 46), 'str', 'Even Coefficients Equal Zero')
        # Processing the call keyword arguments (line 387)
        kwargs_316151 = {}
        # Getting the type of 'assert_' (line 387)
        assert__316135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 387)
        assert__call_result_316152 = invoke(stypy.reporting.localization.Localization(__file__, 387, 8), assert__316135, *[all_call_result_316149, str_316150], **kwargs_316151)
        
        
        # Assigning a Call to a Tuple (line 390):
        
        # Assigning a Subscript to a Name (line 390):
        
        # Obtaining the type of the subscript
        int_316153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 8), 'int')
        
        # Call to freqz(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'h' (line 390)
        h_316155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 21), 'h', False)
        int_316156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 24), 'int')
        # Processing the call keyword arguments (line 390)
        kwargs_316157 = {}
        # Getting the type of 'freqz' (line 390)
        freqz_316154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 390)
        freqz_call_result_316158 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), freqz_316154, *[h_316155, int_316156], **kwargs_316157)
        
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___316159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), freqz_call_result_316158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_316160 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), getitem___316159, int_316153)
        
        # Assigning a type to the variable 'tuple_var_assignment_313941' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_313941', subscript_call_result_316160)
        
        # Assigning a Subscript to a Name (line 390):
        
        # Obtaining the type of the subscript
        int_316161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 8), 'int')
        
        # Call to freqz(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'h' (line 390)
        h_316163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 21), 'h', False)
        int_316164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 24), 'int')
        # Processing the call keyword arguments (line 390)
        kwargs_316165 = {}
        # Getting the type of 'freqz' (line 390)
        freqz_316162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 390)
        freqz_call_result_316166 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), freqz_316162, *[h_316163, int_316164], **kwargs_316165)
        
        # Obtaining the member '__getitem__' of a type (line 390)
        getitem___316167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), freqz_call_result_316166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 390)
        subscript_call_result_316168 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), getitem___316167, int_316161)
        
        # Assigning a type to the variable 'tuple_var_assignment_313942' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_313942', subscript_call_result_316168)
        
        # Assigning a Name to a Name (line 390):
        # Getting the type of 'tuple_var_assignment_313941' (line 390)
        tuple_var_assignment_313941_316169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_313941')
        # Assigning a type to the variable 'w' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'w', tuple_var_assignment_313941_316169)
        
        # Assigning a Name to a Name (line 390):
        # Getting the type of 'tuple_var_assignment_313942' (line 390)
        tuple_var_assignment_313942_316170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'tuple_var_assignment_313942')
        # Assigning a type to the variable 'H' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 11), 'H', tuple_var_assignment_313942_316170)
        
        # Assigning a BinOp to a Name (line 391):
        
        # Assigning a BinOp to a Name (line 391):
        # Getting the type of 'w' (line 391)
        w_316171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'w')
        int_316172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 14), 'int')
        # Applying the binary operator 'div' (line 391)
        result_div_316173 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 12), 'div', w_316171, int_316172)
        
        # Getting the type of 'np' (line 391)
        np_316174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'np')
        # Obtaining the member 'pi' of a type (line 391)
        pi_316175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), np_316174, 'pi')
        # Applying the binary operator 'div' (line 391)
        result_div_316176 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 15), 'div', result_div_316173, pi_316175)
        
        # Assigning a type to the variable 'f' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'f', result_div_316176)
        
        # Assigning a Call to a Name (line 392):
        
        # Assigning a Call to a Name (line 392):
        
        # Call to abs(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'H' (line 392)
        H_316178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'H', False)
        # Processing the call keyword arguments (line 392)
        kwargs_316179 = {}
        # Getting the type of 'abs' (line 392)
        abs_316177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 392)
        abs_call_result_316180 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), abs_316177, *[H_316178], **kwargs_316179)
        
        # Assigning a type to the variable 'Hmag' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'Hmag', abs_call_result_316180)
        
        # Call to assert_(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Call to all(...): (line 395)
        # Processing the call keyword arguments (line 395)
        kwargs_316191 = {}
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_316182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        int_316183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 22), list_316182, int_316183)
        # Adding element type (line 395)
        int_316184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 22), list_316182, int_316184)
        
        # Getting the type of 'Hmag' (line 395)
        Hmag_316185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 17), 'Hmag', False)
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___316186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 17), Hmag_316185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_316187 = invoke(stypy.reporting.localization.Localization(__file__, 395, 17), getitem___316186, list_316182)
        
        float_316188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 33), 'float')
        # Applying the binary operator '<' (line 395)
        result_lt_316189 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 17), '<', subscript_call_result_316187, float_316188)
        
        # Obtaining the member 'all' of a type (line 395)
        all_316190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 17), result_lt_316189, 'all')
        # Calling all(args, kwargs) (line 395)
        all_call_result_316192 = invoke(stypy.reporting.localization.Localization(__file__, 395, 17), all_316190, *[], **kwargs_316191)
        
        str_316193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 46), 'str', 'Zero at zero and pi')
        # Processing the call keyword arguments (line 395)
        kwargs_316194 = {}
        # Getting the type of 'assert_' (line 395)
        assert__316181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 395)
        assert__call_result_316195 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__316181, *[all_call_result_316192, str_316193], **kwargs_316194)
        
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to logical_and(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Getting the type of 'f' (line 398)
        f_316198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 29), 'f', False)
        # Getting the type of 'a' (line 398)
        a_316199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 33), 'a', False)
        # Applying the binary operator '>' (line 398)
        result_gt_316200 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 29), '>', f_316198, a_316199)
        
        
        # Getting the type of 'f' (line 398)
        f_316201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 36), 'f', False)
        float_316202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 40), 'float')
        # Getting the type of 'a' (line 398)
        a_316203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 44), 'a', False)
        # Applying the binary operator '-' (line 398)
        result_sub_316204 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 40), '-', float_316202, a_316203)
        
        # Applying the binary operator '<' (line 398)
        result_lt_316205 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 36), '<', f_316201, result_sub_316204)
        
        # Processing the call keyword arguments (line 398)
        kwargs_316206 = {}
        # Getting the type of 'np' (line 398)
        np_316196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 14), 'np', False)
        # Obtaining the member 'logical_and' of a type (line 398)
        logical_and_316197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 14), np_316196, 'logical_and')
        # Calling logical_and(args, kwargs) (line 398)
        logical_and_call_result_316207 = invoke(stypy.reporting.localization.Localization(__file__, 398, 14), logical_and_316197, *[result_gt_316200, result_lt_316205], **kwargs_316206)
        
        # Assigning a type to the variable 'idx' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'idx', logical_and_call_result_316207)
        
        # Call to assert_(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to all(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_316221 = {}
        
        
        # Call to abs(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 399)
        idx_316210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 26), 'idx', False)
        # Getting the type of 'Hmag' (line 399)
        Hmag_316211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'Hmag', False)
        # Obtaining the member '__getitem__' of a type (line 399)
        getitem___316212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 21), Hmag_316211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 399)
        subscript_call_result_316213 = invoke(stypy.reporting.localization.Localization(__file__, 399, 21), getitem___316212, idx_316210)
        
        int_316214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 33), 'int')
        # Applying the binary operator '-' (line 399)
        result_sub_316215 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 21), '-', subscript_call_result_316213, int_316214)
        
        # Processing the call keyword arguments (line 399)
        kwargs_316216 = {}
        # Getting the type of 'abs' (line 399)
        abs_316209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'abs', False)
        # Calling abs(args, kwargs) (line 399)
        abs_call_result_316217 = invoke(stypy.reporting.localization.Localization(__file__, 399, 17), abs_316209, *[result_sub_316215], **kwargs_316216)
        
        float_316218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 38), 'float')
        # Applying the binary operator '<' (line 399)
        result_lt_316219 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 17), '<', abs_call_result_316217, float_316218)
        
        # Obtaining the member 'all' of a type (line 399)
        all_316220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 17), result_lt_316219, 'all')
        # Calling all(args, kwargs) (line 399)
        all_call_result_316222 = invoke(stypy.reporting.localization.Localization(__file__, 399, 17), all_316220, *[], **kwargs_316221)
        
        str_316223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 52), 'str', 'Pass Band Close To Unity')
        # Processing the call keyword arguments (line 399)
        kwargs_316224 = {}
        # Getting the type of 'assert_' (line 399)
        assert__316208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 399)
        assert__call_result_316225 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), assert__316208, *[all_call_result_316222, str_316223], **kwargs_316224)
        
        
        # ################# End of 'test_hilbert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hilbert' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_316226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hilbert'
        return stypy_return_type_316226


    @norecursion
    def test_compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_compare'
        module_type_store = module_type_store.open_function_context('test_compare', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRemez.test_compare.__dict__.__setitem__('stypy_localization', localization)
        TestRemez.test_compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRemez.test_compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRemez.test_compare.__dict__.__setitem__('stypy_function_name', 'TestRemez.test_compare')
        TestRemez.test_compare.__dict__.__setitem__('stypy_param_names_list', [])
        TestRemez.test_compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRemez.test_compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRemez.test_compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRemez.test_compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRemez.test_compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRemez.test_compare.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRemez.test_compare', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_compare', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_compare(...)' code ##################

        
        # Assigning a List to a Name (line 403):
        
        # Assigning a List to a Name (line 403):
        
        # Obtaining an instance of the builtin type 'list' (line 403)
        list_316227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 403)
        # Adding element type (line 403)
        float_316228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316228)
        # Adding element type (line 403)
        float_316229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316229)
        # Adding element type (line 403)
        float_316230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316230)
        # Adding element type (line 403)
        float_316231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316231)
        # Adding element type (line 403)
        float_316232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316232)
        # Adding element type (line 403)
        float_316233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316233)
        # Adding element type (line 403)
        float_316234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316234)
        # Adding element type (line 403)
        float_316235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316235)
        # Adding element type (line 403)
        float_316236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316236)
        # Adding element type (line 403)
        float_316237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316237)
        # Adding element type (line 403)
        float_316238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316238)
        # Adding element type (line 403)
        float_316239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 12), list_316227, float_316239)
        
        # Assigning a type to the variable 'k' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'k', list_316227)
        
        # Assigning a Call to a Name (line 407):
        
        # Assigning a Call to a Name (line 407):
        
        # Call to remez(...): (line 407)
        # Processing the call arguments (line 407)
        int_316241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_316242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        int_316243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 22), list_316242, int_316243)
        # Adding element type (line 407)
        float_316244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 22), list_316242, float_316244)
        # Adding element type (line 407)
        float_316245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 22), list_316242, float_316245)
        # Adding element type (line 407)
        int_316246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 22), list_316242, int_316246)
        
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_316247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        int_316248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 40), list_316247, int_316248)
        # Adding element type (line 407)
        int_316249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 40), list_316247, int_316249)
        
        # Processing the call keyword arguments (line 407)
        float_316250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 51), 'float')
        keyword_316251 = float_316250
        kwargs_316252 = {'Hz': keyword_316251}
        # Getting the type of 'remez' (line 407)
        remez_316240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 12), 'remez', False)
        # Calling remez(args, kwargs) (line 407)
        remez_call_result_316253 = invoke(stypy.reporting.localization.Localization(__file__, 407, 12), remez_316240, *[int_316241, list_316242, list_316247], **kwargs_316252)
        
        # Assigning a type to the variable 'h' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'h', remez_call_result_316253)
        
        # Call to assert_allclose(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'h' (line 408)
        h_316255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'h', False)
        # Getting the type of 'k' (line 408)
        k_316256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 27), 'k', False)
        # Processing the call keyword arguments (line 408)
        kwargs_316257 = {}
        # Getting the type of 'assert_allclose' (line 408)
        assert_allclose_316254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 408)
        assert_allclose_call_result_316258 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), assert_allclose_316254, *[h_316255, k_316256], **kwargs_316257)
        
        
        # Assigning a Call to a Name (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to remez(...): (line 409)
        # Processing the call arguments (line 409)
        int_316260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_316261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        int_316262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_316261, int_316262)
        # Adding element type (line 409)
        float_316263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_316261, float_316263)
        # Adding element type (line 409)
        float_316264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_316261, float_316264)
        # Adding element type (line 409)
        int_316265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_316261, int_316265)
        
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_316266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        int_316267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 40), list_316266, int_316267)
        # Adding element type (line 409)
        int_316268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 40), list_316266, int_316268)
        
        # Processing the call keyword arguments (line 409)
        float_316269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 51), 'float')
        keyword_316270 = float_316269
        kwargs_316271 = {'fs': keyword_316270}
        # Getting the type of 'remez' (line 409)
        remez_316259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'remez', False)
        # Calling remez(args, kwargs) (line 409)
        remez_call_result_316272 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), remez_316259, *[int_316260, list_316261, list_316266], **kwargs_316271)
        
        # Assigning a type to the variable 'h' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'h', remez_call_result_316272)
        
        # Call to assert_allclose(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'h' (line 410)
        h_316274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 24), 'h', False)
        # Getting the type of 'k' (line 410)
        k_316275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 27), 'k', False)
        # Processing the call keyword arguments (line 410)
        kwargs_316276 = {}
        # Getting the type of 'assert_allclose' (line 410)
        assert_allclose_316273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 410)
        assert_allclose_call_result_316277 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), assert_allclose_316273, *[h_316274, k_316275], **kwargs_316276)
        
        
        # Assigning a List to a Name (line 412):
        
        # Assigning a List to a Name (line 412):
        
        # Obtaining an instance of the builtin type 'list' (line 412)
        list_316278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 412)
        # Adding element type (line 412)
        float_316279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316279)
        # Adding element type (line 412)
        float_316280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316280)
        # Adding element type (line 412)
        float_316281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316281)
        # Adding element type (line 412)
        float_316282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316282)
        # Adding element type (line 412)
        float_316283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316283)
        # Adding element type (line 412)
        float_316284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316284)
        # Adding element type (line 412)
        float_316285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316285)
        # Adding element type (line 412)
        float_316286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316286)
        # Adding element type (line 412)
        float_316287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316287)
        # Adding element type (line 412)
        float_316288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316288)
        # Adding element type (line 412)
        float_316289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316289)
        # Adding element type (line 412)
        float_316290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316290)
        # Adding element type (line 412)
        float_316291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316291)
        # Adding element type (line 412)
        float_316292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316292)
        # Adding element type (line 412)
        float_316293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316293)
        # Adding element type (line 412)
        float_316294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316294)
        # Adding element type (line 412)
        float_316295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316295)
        # Adding element type (line 412)
        float_316296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316296)
        # Adding element type (line 412)
        float_316297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316297)
        # Adding element type (line 412)
        float_316298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316298)
        # Adding element type (line 412)
        float_316299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 412, 12), list_316278, float_316299)
        
        # Assigning a type to the variable 'h' (line 412)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'h', list_316278)
        
        # Call to assert_allclose(...): (line 419)
        # Processing the call arguments (line 419)
        
        # Call to remez(...): (line 419)
        # Processing the call arguments (line 419)
        int_316302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 30), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 419)
        list_316303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 419)
        # Adding element type (line 419)
        int_316304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 34), list_316303, int_316304)
        # Adding element type (line 419)
        float_316305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 34), list_316303, float_316305)
        # Adding element type (line 419)
        float_316306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 34), list_316303, float_316306)
        # Adding element type (line 419)
        int_316307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 34), list_316303, int_316307)
        
        
        # Obtaining an instance of the builtin type 'list' (line 419)
        list_316308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 419)
        # Adding element type (line 419)
        int_316309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 52), list_316308, int_316309)
        # Adding element type (line 419)
        int_316310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 52), list_316308, int_316310)
        
        # Processing the call keyword arguments (line 419)
        float_316311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 63), 'float')
        keyword_316312 = float_316311
        kwargs_316313 = {'Hz': keyword_316312}
        # Getting the type of 'remez' (line 419)
        remez_316301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'remez', False)
        # Calling remez(args, kwargs) (line 419)
        remez_call_result_316314 = invoke(stypy.reporting.localization.Localization(__file__, 419, 24), remez_316301, *[int_316302, list_316303, list_316308], **kwargs_316313)
        
        # Getting the type of 'h' (line 419)
        h_316315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 68), 'h', False)
        # Processing the call keyword arguments (line 419)
        kwargs_316316 = {}
        # Getting the type of 'assert_allclose' (line 419)
        assert_allclose_316300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 419)
        assert_allclose_call_result_316317 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), assert_allclose_316300, *[remez_call_result_316314, h_316315], **kwargs_316316)
        
        
        # Call to assert_allclose(...): (line 420)
        # Processing the call arguments (line 420)
        
        # Call to remez(...): (line 420)
        # Processing the call arguments (line 420)
        int_316320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 30), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 420)
        list_316321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 420)
        # Adding element type (line 420)
        int_316322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 34), list_316321, int_316322)
        # Adding element type (line 420)
        float_316323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 34), list_316321, float_316323)
        # Adding element type (line 420)
        float_316324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 34), list_316321, float_316324)
        # Adding element type (line 420)
        int_316325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 34), list_316321, int_316325)
        
        
        # Obtaining an instance of the builtin type 'list' (line 420)
        list_316326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 420)
        # Adding element type (line 420)
        int_316327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 52), list_316326, int_316327)
        # Adding element type (line 420)
        int_316328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 52), list_316326, int_316328)
        
        # Processing the call keyword arguments (line 420)
        float_316329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 63), 'float')
        keyword_316330 = float_316329
        kwargs_316331 = {'fs': keyword_316330}
        # Getting the type of 'remez' (line 420)
        remez_316319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'remez', False)
        # Calling remez(args, kwargs) (line 420)
        remez_call_result_316332 = invoke(stypy.reporting.localization.Localization(__file__, 420, 24), remez_316319, *[int_316320, list_316321, list_316326], **kwargs_316331)
        
        # Getting the type of 'h' (line 420)
        h_316333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 68), 'h', False)
        # Processing the call keyword arguments (line 420)
        kwargs_316334 = {}
        # Getting the type of 'assert_allclose' (line 420)
        assert_allclose_316318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 420)
        assert_allclose_call_result_316335 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), assert_allclose_316318, *[remez_call_result_316332, h_316333], **kwargs_316334)
        
        
        # ################# End of 'test_compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_compare' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_316336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_compare'
        return stypy_return_type_316336


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRemez.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRemez' (line 367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'TestRemez', TestRemez)
# Declaration of the 'TestFirls' class

class TestFirls(object, ):

    @norecursion
    def test_bad_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_args'
        module_type_store = module_type_store.open_function_context('test_bad_args', 425, 4, False)
        # Assigning a type to the variable 'self' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_localization', localization)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_function_name', 'TestFirls.test_bad_args')
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirls.test_bad_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirls.test_bad_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_args(...)' code ##################

        
        # Call to assert_raises(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'ValueError' (line 427)
        ValueError_316338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 427)
        firls_316339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'firls', False)
        int_316340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 427)
        list_316341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 427)
        # Adding element type (line 427)
        float_316342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 45), list_316341, float_316342)
        # Adding element type (line 427)
        float_316343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 45), list_316341, float_316343)
        
        
        # Obtaining an instance of the builtin type 'list' (line 427)
        list_316344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 427)
        # Adding element type (line 427)
        int_316345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 57), list_316344, int_316345)
        # Adding element type (line 427)
        int_316346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 427, 57), list_316344, int_316346)
        
        # Processing the call keyword arguments (line 427)
        kwargs_316347 = {}
        # Getting the type of 'assert_raises' (line 427)
        assert_raises_316337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 427)
        assert_raises_call_result_316348 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert_raises_316337, *[ValueError_316338, firls_316339, int_316340, list_316341, list_316344], **kwargs_316347)
        
        
        # Call to assert_raises(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'ValueError' (line 429)
        ValueError_316350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 429)
        firls_316351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 34), 'firls', False)
        int_316352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 429)
        list_316353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 429)
        # Adding element type (line 429)
        float_316354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 45), list_316353, float_316354)
        # Adding element type (line 429)
        float_316355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 45), list_316353, float_316355)
        # Adding element type (line 429)
        float_316356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 45), list_316353, float_316356)
        
        
        # Obtaining an instance of the builtin type 'list' (line 429)
        list_316357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 429)
        # Adding element type (line 429)
        int_316358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 62), list_316357, int_316358)
        # Adding element type (line 429)
        int_316359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 62), list_316357, int_316359)
        # Adding element type (line 429)
        int_316360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 62), list_316357, int_316360)
        
        # Processing the call keyword arguments (line 429)
        kwargs_316361 = {}
        # Getting the type of 'assert_raises' (line 429)
        assert_raises_316349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 429)
        assert_raises_call_result_316362 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), assert_raises_316349, *[ValueError_316350, firls_316351, int_316352, list_316353, list_316357], **kwargs_316361)
        
        
        # Call to assert_raises(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'ValueError' (line 431)
        ValueError_316364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 431)
        firls_316365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 34), 'firls', False)
        int_316366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_316367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        float_316368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 45), list_316367, float_316368)
        # Adding element type (line 431)
        float_316369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 45), list_316367, float_316369)
        # Adding element type (line 431)
        float_316370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 45), list_316367, float_316370)
        # Adding element type (line 431)
        float_316371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 45), list_316367, float_316371)
        
        
        # Obtaining an instance of the builtin type 'list' (line 431)
        list_316372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 431)
        # Adding element type (line 431)
        int_316373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 67), list_316372, int_316373)
        # Adding element type (line 431)
        int_316374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 71), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 67), list_316372, int_316374)
        # Adding element type (line 431)
        int_316375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 74), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 67), list_316372, int_316375)
        
        # Processing the call keyword arguments (line 431)
        kwargs_316376 = {}
        # Getting the type of 'assert_raises' (line 431)
        assert_raises_316363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 431)
        assert_raises_call_result_316377 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), assert_raises_316363, *[ValueError_316364, firls_316365, int_316366, list_316367, list_316372], **kwargs_316376)
        
        
        # Call to assert_raises(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'ValueError' (line 433)
        ValueError_316379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 433)
        firls_316380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 34), 'firls', False)
        int_316381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_316382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        float_316383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 45), list_316382, float_316383)
        # Adding element type (line 433)
        float_316384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 45), list_316382, float_316384)
        
        
        # Obtaining an instance of the builtin type 'list' (line 433)
        list_316385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 433)
        # Adding element type (line 433)
        int_316386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 57), list_316385, int_316386)
        # Adding element type (line 433)
        int_316387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 57), list_316385, int_316387)
        
        # Processing the call keyword arguments (line 433)
        kwargs_316388 = {}
        # Getting the type of 'assert_raises' (line 433)
        assert_raises_316378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 433)
        assert_raises_call_result_316389 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), assert_raises_316378, *[ValueError_316379, firls_316380, int_316381, list_316382, list_316385], **kwargs_316388)
        
        
        # Call to assert_raises(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'ValueError' (line 434)
        ValueError_316391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 434)
        firls_316392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 34), 'firls', False)
        int_316393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 434)
        list_316394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 434)
        # Adding element type (line 434)
        float_316395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 45), list_316394, float_316395)
        # Adding element type (line 434)
        float_316396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 45), list_316394, float_316396)
        # Adding element type (line 434)
        float_316397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 45), list_316394, float_316397)
        # Adding element type (line 434)
        float_316398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 45), list_316394, float_316398)
        
        
        # Obtaining an instance of the builtin type 'list' (line 434)
        list_316399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 434)
        # Adding element type (line 434)
        int_316400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 67), list_316399, int_316400)
        
        int_316401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 73), 'int')
        # Applying the binary operator '*' (line 434)
        result_mul_316402 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 67), '*', list_316399, int_316401)
        
        # Processing the call keyword arguments (line 434)
        kwargs_316403 = {}
        # Getting the type of 'assert_raises' (line 434)
        assert_raises_316390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 434)
        assert_raises_call_result_316404 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), assert_raises_316390, *[ValueError_316391, firls_316392, int_316393, list_316394, result_mul_316402], **kwargs_316403)
        
        
        # Call to assert_raises(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'ValueError' (line 435)
        ValueError_316406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 435)
        firls_316407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 34), 'firls', False)
        int_316408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_316409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        float_316410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 45), list_316409, float_316410)
        # Adding element type (line 435)
        float_316411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 45), list_316409, float_316411)
        # Adding element type (line 435)
        float_316412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 45), list_316409, float_316412)
        # Adding element type (line 435)
        float_316413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 45), list_316409, float_316413)
        
        
        # Obtaining an instance of the builtin type 'list' (line 435)
        list_316414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 435)
        # Adding element type (line 435)
        int_316415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 67), list_316414, int_316415)
        
        int_316416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 73), 'int')
        # Applying the binary operator '*' (line 435)
        result_mul_316417 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 67), '*', list_316414, int_316416)
        
        # Processing the call keyword arguments (line 435)
        kwargs_316418 = {}
        # Getting the type of 'assert_raises' (line 435)
        assert_raises_316405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 435)
        assert_raises_call_result_316419 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), assert_raises_316405, *[ValueError_316406, firls_316407, int_316408, list_316409, result_mul_316417], **kwargs_316418)
        
        
        # Call to assert_raises(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'ValueError' (line 436)
        ValueError_316421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 436)
        firls_316422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 34), 'firls', False)
        int_316423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 436)
        list_316424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 436)
        # Adding element type (line 436)
        float_316425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 45), list_316424, float_316425)
        # Adding element type (line 436)
        float_316426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 45), list_316424, float_316426)
        # Adding element type (line 436)
        float_316427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 45), list_316424, float_316427)
        # Adding element type (line 436)
        float_316428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 45), list_316424, float_316428)
        
        
        # Obtaining an instance of the builtin type 'list' (line 436)
        list_316429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 67), 'list')
        # Adding type elements to the builtin type 'list' instance (line 436)
        # Adding element type (line 436)
        int_316430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 67), list_316429, int_316430)
        
        int_316431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 73), 'int')
        # Applying the binary operator '*' (line 436)
        result_mul_316432 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 67), '*', list_316429, int_316431)
        
        # Processing the call keyword arguments (line 436)
        kwargs_316433 = {}
        # Getting the type of 'assert_raises' (line 436)
        assert_raises_316420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 436)
        assert_raises_call_result_316434 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), assert_raises_316420, *[ValueError_316421, firls_316422, int_316423, list_316424, result_mul_316432], **kwargs_316433)
        
        
        # Call to assert_raises(...): (line 438)
        # Processing the call arguments (line 438)
        # Getting the type of 'ValueError' (line 438)
        ValueError_316436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 438)
        firls_316437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 34), 'firls', False)
        int_316438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 438)
        list_316439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 438)
        # Adding element type (line 438)
        float_316440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 45), list_316439, float_316440)
        # Adding element type (line 438)
        float_316441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 45), list_316439, float_316441)
        
        
        # Obtaining an instance of the builtin type 'list' (line 438)
        list_316442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 438)
        # Adding element type (line 438)
        int_316443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 57), list_316442, int_316443)
        # Adding element type (line 438)
        int_316444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 438, 57), list_316442, int_316444)
        
        # Processing the call keyword arguments (line 438)
        kwargs_316445 = {}
        # Getting the type of 'assert_raises' (line 438)
        assert_raises_316435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 438)
        assert_raises_call_result_316446 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), assert_raises_316435, *[ValueError_316436, firls_316437, int_316438, list_316439, list_316442], **kwargs_316445)
        
        
        # Call to assert_raises(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'ValueError' (line 440)
        ValueError_316448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 440)
        firls_316449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 34), 'firls', False)
        int_316450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 440)
        list_316451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 440)
        # Adding element type (line 440)
        float_316452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 45), list_316451, float_316452)
        # Adding element type (line 440)
        float_316453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 45), list_316451, float_316453)
        
        
        # Obtaining an instance of the builtin type 'list' (line 440)
        list_316454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 440)
        # Adding element type (line 440)
        int_316455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 57), list_316454, int_316455)
        # Adding element type (line 440)
        int_316456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 57), list_316454, int_316456)
        
        
        # Obtaining an instance of the builtin type 'list' (line 440)
        list_316457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 440)
        # Adding element type (line 440)
        int_316458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 65), list_316457, int_316458)
        # Adding element type (line 440)
        int_316459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 65), list_316457, int_316459)
        
        # Processing the call keyword arguments (line 440)
        kwargs_316460 = {}
        # Getting the type of 'assert_raises' (line 440)
        assert_raises_316447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 440)
        assert_raises_call_result_316461 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), assert_raises_316447, *[ValueError_316448, firls_316449, int_316450, list_316451, list_316454, list_316457], **kwargs_316460)
        
        
        # Call to assert_raises(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'ValueError' (line 442)
        ValueError_316463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'ValueError', False)
        # Getting the type of 'firls' (line 442)
        firls_316464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 34), 'firls', False)
        int_316465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 41), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 442)
        list_316466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 442)
        # Adding element type (line 442)
        float_316467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 45), list_316466, float_316467)
        # Adding element type (line 442)
        float_316468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 45), list_316466, float_316468)
        
        
        # Obtaining an instance of the builtin type 'list' (line 442)
        list_316469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 442)
        # Adding element type (line 442)
        int_316470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 57), list_316469, int_316470)
        # Adding element type (line 442)
        int_316471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 57), list_316469, int_316471)
        
        
        # Obtaining an instance of the builtin type 'list' (line 442)
        list_316472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 442)
        # Adding element type (line 442)
        int_316473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 65), list_316472, int_316473)
        
        # Processing the call keyword arguments (line 442)
        kwargs_316474 = {}
        # Getting the type of 'assert_raises' (line 442)
        assert_raises_316462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 442)
        assert_raises_call_result_316475 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), assert_raises_316462, *[ValueError_316463, firls_316464, int_316465, list_316466, list_316469, list_316472], **kwargs_316474)
        
        
        # ################# End of 'test_bad_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_args' in the type store
        # Getting the type of 'stypy_return_type' (line 425)
        stypy_return_type_316476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316476)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_args'
        return stypy_return_type_316476


    @norecursion
    def test_firls(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_firls'
        module_type_store = module_type_store.open_function_context('test_firls', 444, 4, False)
        # Assigning a type to the variable 'self' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirls.test_firls.__dict__.__setitem__('stypy_localization', localization)
        TestFirls.test_firls.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirls.test_firls.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirls.test_firls.__dict__.__setitem__('stypy_function_name', 'TestFirls.test_firls')
        TestFirls.test_firls.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirls.test_firls.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirls.test_firls.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirls.test_firls.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirls.test_firls.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirls.test_firls.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirls.test_firls.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirls.test_firls', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_firls', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_firls(...)' code ##################

        
        # Assigning a Num to a Name (line 445):
        
        # Assigning a Num to a Name (line 445):
        int_316477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'int')
        # Assigning a type to the variable 'N' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'N', int_316477)
        
        # Assigning a Num to a Name (line 446):
        
        # Assigning a Num to a Name (line 446):
        float_316478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 12), 'float')
        # Assigning a type to the variable 'a' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'a', float_316478)
        
        # Assigning a Call to a Name (line 449):
        
        # Assigning a Call to a Name (line 449):
        
        # Call to firls(...): (line 449)
        # Processing the call arguments (line 449)
        int_316480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 449)
        list_316481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 449)
        # Adding element type (line 449)
        int_316482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 22), list_316481, int_316482)
        # Adding element type (line 449)
        # Getting the type of 'a' (line 449)
        a_316483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 26), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 22), list_316481, a_316483)
        # Adding element type (line 449)
        float_316484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 29), 'float')
        # Getting the type of 'a' (line 449)
        a_316485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 33), 'a', False)
        # Applying the binary operator '-' (line 449)
        result_sub_316486 = python_operator(stypy.reporting.localization.Localization(__file__, 449, 29), '-', float_316484, a_316485)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 22), list_316481, result_sub_316486)
        # Adding element type (line 449)
        float_316487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 22), list_316481, float_316487)
        
        
        # Obtaining an instance of the builtin type 'list' (line 449)
        list_316488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 449)
        # Adding element type (line 449)
        int_316489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 42), list_316488, int_316489)
        # Adding element type (line 449)
        int_316490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 42), list_316488, int_316490)
        # Adding element type (line 449)
        int_316491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 42), list_316488, int_316491)
        # Adding element type (line 449)
        int_316492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 42), list_316488, int_316492)
        
        # Processing the call keyword arguments (line 449)
        float_316493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 59), 'float')
        keyword_316494 = float_316493
        kwargs_316495 = {'fs': keyword_316494}
        # Getting the type of 'firls' (line 449)
        firls_316479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'firls', False)
        # Calling firls(args, kwargs) (line 449)
        firls_call_result_316496 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), firls_316479, *[int_316480, list_316481, list_316488], **kwargs_316495)
        
        # Assigning a type to the variable 'h' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'h', firls_call_result_316496)
        
        # Call to assert_equal(...): (line 452)
        # Processing the call arguments (line 452)
        
        # Call to len(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'h' (line 452)
        h_316499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 25), 'h', False)
        # Processing the call keyword arguments (line 452)
        kwargs_316500 = {}
        # Getting the type of 'len' (line 452)
        len_316498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 21), 'len', False)
        # Calling len(args, kwargs) (line 452)
        len_call_result_316501 = invoke(stypy.reporting.localization.Localization(__file__, 452, 21), len_316498, *[h_316499], **kwargs_316500)
        
        # Getting the type of 'N' (line 452)
        N_316502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 29), 'N', False)
        # Processing the call keyword arguments (line 452)
        kwargs_316503 = {}
        # Getting the type of 'assert_equal' (line 452)
        assert_equal_316497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 452)
        assert_equal_call_result_316504 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), assert_equal_316497, *[len_call_result_316501, N_316502], **kwargs_316503)
        
        
        # Assigning a BinOp to a Name (line 455):
        
        # Assigning a BinOp to a Name (line 455):
        # Getting the type of 'N' (line 455)
        N_316505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'N')
        int_316506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 18), 'int')
        # Applying the binary operator '-' (line 455)
        result_sub_316507 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 16), '-', N_316505, int_316506)
        
        int_316508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 24), 'int')
        # Applying the binary operator '//' (line 455)
        result_floordiv_316509 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 15), '//', result_sub_316507, int_316508)
        
        # Assigning a type to the variable 'midx' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'midx', result_floordiv_316509)
        
        # Call to assert_array_almost_equal(...): (line 456)
        # Processing the call arguments (line 456)
        
        # Obtaining the type of the subscript
        # Getting the type of 'midx' (line 456)
        midx_316511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 37), 'midx', False)
        slice_316512 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 34), None, midx_316511, None)
        # Getting the type of 'h' (line 456)
        h_316513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 34), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 456)
        getitem___316514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 34), h_316513, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 456)
        subscript_call_result_316515 = invoke(stypy.reporting.localization.Localization(__file__, 456, 34), getitem___316514, slice_316512)
        
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'midx' (line 456)
        midx_316516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 48), 'midx', False)
        # Applying the 'usub' unary operator (line 456)
        result___neg___316517 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 47), 'usub', midx_316516)
        
        int_316518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 53), 'int')
        # Applying the binary operator '-' (line 456)
        result_sub_316519 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 47), '-', result___neg___316517, int_316518)
        
        int_316520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 55), 'int')
        slice_316521 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 456, 44), None, result_sub_316519, int_316520)
        # Getting the type of 'h' (line 456)
        h_316522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 456)
        getitem___316523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 44), h_316522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 456)
        subscript_call_result_316524 = invoke(stypy.reporting.localization.Localization(__file__, 456, 44), getitem___316523, slice_316521)
        
        # Processing the call keyword arguments (line 456)
        kwargs_316525 = {}
        # Getting the type of 'assert_array_almost_equal' (line 456)
        assert_array_almost_equal_316510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 456)
        assert_array_almost_equal_call_result_316526 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), assert_array_almost_equal_316510, *[subscript_call_result_316515, subscript_call_result_316524], **kwargs_316525)
        
        
        # Call to assert_almost_equal(...): (line 459)
        # Processing the call arguments (line 459)
        
        # Obtaining the type of the subscript
        # Getting the type of 'midx' (line 459)
        midx_316528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 30), 'midx', False)
        # Getting the type of 'h' (line 459)
        h_316529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 28), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 459)
        getitem___316530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 28), h_316529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 459)
        subscript_call_result_316531 = invoke(stypy.reporting.localization.Localization(__file__, 459, 28), getitem___316530, midx_316528)
        
        float_316532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 37), 'float')
        # Processing the call keyword arguments (line 459)
        kwargs_316533 = {}
        # Getting the type of 'assert_almost_equal' (line 459)
        assert_almost_equal_316527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 459)
        assert_almost_equal_call_result_316534 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), assert_almost_equal_316527, *[subscript_call_result_316531, float_316532], **kwargs_316533)
        
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to hstack(...): (line 463)
        # Processing the call arguments (line 463)
        
        # Obtaining an instance of the builtin type 'tuple' (line 463)
        tuple_316537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 463)
        # Adding element type (line 463)
        
        # Obtaining the type of the subscript
        int_316538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 28), 'int')
        # Getting the type of 'midx' (line 463)
        midx_316539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 30), 'midx', False)
        int_316540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 35), 'int')
        slice_316541 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 26), int_316538, midx_316539, int_316540)
        # Getting the type of 'h' (line 463)
        h_316542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 26), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___316543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 26), h_316542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_316544 = invoke(stypy.reporting.localization.Localization(__file__, 463, 26), getitem___316543, slice_316541)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 26), tuple_316537, subscript_call_result_316544)
        # Adding element type (line 463)
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'midx' (line 463)
        midx_316545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 42), 'midx', False)
        # Applying the 'usub' unary operator (line 463)
        result___neg___316546 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 41), 'usub', midx_316545)
        
        int_316547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 47), 'int')
        # Applying the binary operator '+' (line 463)
        result_add_316548 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 41), '+', result___neg___316546, int_316547)
        
        int_316549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 50), 'int')
        slice_316550 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 463, 39), result_add_316548, None, int_316549)
        # Getting the type of 'h' (line 463)
        h_316551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 39), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 463)
        getitem___316552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 39), h_316551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 463)
        subscript_call_result_316553 = invoke(stypy.reporting.localization.Localization(__file__, 463, 39), getitem___316552, slice_316550)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 26), tuple_316537, subscript_call_result_316553)
        
        # Processing the call keyword arguments (line 463)
        kwargs_316554 = {}
        # Getting the type of 'np' (line 463)
        np_316535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 463)
        hstack_316536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 15), np_316535, 'hstack')
        # Calling hstack(args, kwargs) (line 463)
        hstack_call_result_316555 = invoke(stypy.reporting.localization.Localization(__file__, 463, 15), hstack_316536, *[tuple_316537], **kwargs_316554)
        
        # Assigning a type to the variable 'hodd' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'hodd', hstack_call_result_316555)
        
        # Call to assert_array_almost_equal(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'hodd' (line 464)
        hodd_316557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 34), 'hodd', False)
        int_316558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 40), 'int')
        # Processing the call keyword arguments (line 464)
        kwargs_316559 = {}
        # Getting the type of 'assert_array_almost_equal' (line 464)
        assert_array_almost_equal_316556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 464)
        assert_array_almost_equal_call_result_316560 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), assert_array_almost_equal_316556, *[hodd_316557, int_316558], **kwargs_316559)
        
        
        # Assigning a Call to a Tuple (line 467):
        
        # Assigning a Subscript to a Name (line 467):
        
        # Obtaining the type of the subscript
        int_316561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 8), 'int')
        
        # Call to freqz(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'h' (line 467)
        h_316563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'h', False)
        int_316564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 24), 'int')
        # Processing the call keyword arguments (line 467)
        kwargs_316565 = {}
        # Getting the type of 'freqz' (line 467)
        freqz_316562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 467)
        freqz_call_result_316566 = invoke(stypy.reporting.localization.Localization(__file__, 467, 15), freqz_316562, *[h_316563, int_316564], **kwargs_316565)
        
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___316567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), freqz_call_result_316566, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_316568 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), getitem___316567, int_316561)
        
        # Assigning a type to the variable 'tuple_var_assignment_313943' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'tuple_var_assignment_313943', subscript_call_result_316568)
        
        # Assigning a Subscript to a Name (line 467):
        
        # Obtaining the type of the subscript
        int_316569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 8), 'int')
        
        # Call to freqz(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'h' (line 467)
        h_316571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'h', False)
        int_316572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 24), 'int')
        # Processing the call keyword arguments (line 467)
        kwargs_316573 = {}
        # Getting the type of 'freqz' (line 467)
        freqz_316570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 15), 'freqz', False)
        # Calling freqz(args, kwargs) (line 467)
        freqz_call_result_316574 = invoke(stypy.reporting.localization.Localization(__file__, 467, 15), freqz_316570, *[h_316571, int_316572], **kwargs_316573)
        
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___316575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), freqz_call_result_316574, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_316576 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), getitem___316575, int_316569)
        
        # Assigning a type to the variable 'tuple_var_assignment_313944' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'tuple_var_assignment_313944', subscript_call_result_316576)
        
        # Assigning a Name to a Name (line 467):
        # Getting the type of 'tuple_var_assignment_313943' (line 467)
        tuple_var_assignment_313943_316577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'tuple_var_assignment_313943')
        # Assigning a type to the variable 'w' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'w', tuple_var_assignment_313943_316577)
        
        # Assigning a Name to a Name (line 467):
        # Getting the type of 'tuple_var_assignment_313944' (line 467)
        tuple_var_assignment_313944_316578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'tuple_var_assignment_313944')
        # Assigning a type to the variable 'H' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 11), 'H', tuple_var_assignment_313944_316578)
        
        # Assigning a BinOp to a Name (line 468):
        
        # Assigning a BinOp to a Name (line 468):
        # Getting the type of 'w' (line 468)
        w_316579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'w')
        int_316580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 14), 'int')
        # Applying the binary operator 'div' (line 468)
        result_div_316581 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 12), 'div', w_316579, int_316580)
        
        # Getting the type of 'np' (line 468)
        np_316582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'np')
        # Obtaining the member 'pi' of a type (line 468)
        pi_316583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 16), np_316582, 'pi')
        # Applying the binary operator 'div' (line 468)
        result_div_316584 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 15), 'div', result_div_316581, pi_316583)
        
        # Assigning a type to the variable 'f' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'f', result_div_316584)
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to abs(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'H' (line 469)
        H_316587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 22), 'H', False)
        # Processing the call keyword arguments (line 469)
        kwargs_316588 = {}
        # Getting the type of 'np' (line 469)
        np_316585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 'np', False)
        # Obtaining the member 'abs' of a type (line 469)
        abs_316586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 15), np_316585, 'abs')
        # Calling abs(args, kwargs) (line 469)
        abs_call_result_316589 = invoke(stypy.reporting.localization.Localization(__file__, 469, 15), abs_316586, *[H_316587], **kwargs_316588)
        
        # Assigning a type to the variable 'Hmag' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'Hmag', abs_call_result_316589)
        
        # Assigning a Call to a Name (line 472):
        
        # Assigning a Call to a Name (line 472):
        
        # Call to logical_and(...): (line 472)
        # Processing the call arguments (line 472)
        
        # Getting the type of 'f' (line 472)
        f_316592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 29), 'f', False)
        int_316593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 33), 'int')
        # Applying the binary operator '>' (line 472)
        result_gt_316594 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 29), '>', f_316592, int_316593)
        
        
        # Getting the type of 'f' (line 472)
        f_316595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 36), 'f', False)
        # Getting the type of 'a' (line 472)
        a_316596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'a', False)
        # Applying the binary operator '<' (line 472)
        result_lt_316597 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 36), '<', f_316595, a_316596)
        
        # Processing the call keyword arguments (line 472)
        kwargs_316598 = {}
        # Getting the type of 'np' (line 472)
        np_316590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 14), 'np', False)
        # Obtaining the member 'logical_and' of a type (line 472)
        logical_and_316591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 14), np_316590, 'logical_and')
        # Calling logical_and(args, kwargs) (line 472)
        logical_and_call_result_316599 = invoke(stypy.reporting.localization.Localization(__file__, 472, 14), logical_and_316591, *[result_gt_316594, result_lt_316597], **kwargs_316598)
        
        # Assigning a type to the variable 'idx' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'idx', logical_and_call_result_316599)
        
        # Call to assert_array_almost_equal(...): (line 473)
        # Processing the call arguments (line 473)
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 473)
        idx_316601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 39), 'idx', False)
        # Getting the type of 'Hmag' (line 473)
        Hmag_316602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 34), 'Hmag', False)
        # Obtaining the member '__getitem__' of a type (line 473)
        getitem___316603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 34), Hmag_316602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 473)
        subscript_call_result_316604 = invoke(stypy.reporting.localization.Localization(__file__, 473, 34), getitem___316603, idx_316601)
        
        int_316605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 45), 'int')
        # Processing the call keyword arguments (line 473)
        int_316606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 56), 'int')
        keyword_316607 = int_316606
        kwargs_316608 = {'decimal': keyword_316607}
        # Getting the type of 'assert_array_almost_equal' (line 473)
        assert_array_almost_equal_316600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 473)
        assert_array_almost_equal_call_result_316609 = invoke(stypy.reporting.localization.Localization(__file__, 473, 8), assert_array_almost_equal_316600, *[subscript_call_result_316604, int_316605], **kwargs_316608)
        
        
        # Assigning a Call to a Name (line 476):
        
        # Assigning a Call to a Name (line 476):
        
        # Call to logical_and(...): (line 476)
        # Processing the call arguments (line 476)
        
        # Getting the type of 'f' (line 476)
        f_316612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 29), 'f', False)
        float_316613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 33), 'float')
        # Getting the type of 'a' (line 476)
        a_316614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 37), 'a', False)
        # Applying the binary operator '-' (line 476)
        result_sub_316615 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 33), '-', float_316613, a_316614)
        
        # Applying the binary operator '>' (line 476)
        result_gt_316616 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 29), '>', f_316612, result_sub_316615)
        
        
        # Getting the type of 'f' (line 476)
        f_316617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 40), 'f', False)
        float_316618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 44), 'float')
        # Applying the binary operator '<' (line 476)
        result_lt_316619 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 40), '<', f_316617, float_316618)
        
        # Processing the call keyword arguments (line 476)
        kwargs_316620 = {}
        # Getting the type of 'np' (line 476)
        np_316610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 14), 'np', False)
        # Obtaining the member 'logical_and' of a type (line 476)
        logical_and_316611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 14), np_316610, 'logical_and')
        # Calling logical_and(args, kwargs) (line 476)
        logical_and_call_result_316621 = invoke(stypy.reporting.localization.Localization(__file__, 476, 14), logical_and_316611, *[result_gt_316616, result_lt_316619], **kwargs_316620)
        
        # Assigning a type to the variable 'idx' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'idx', logical_and_call_result_316621)
        
        # Call to assert_array_almost_equal(...): (line 477)
        # Processing the call arguments (line 477)
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 477)
        idx_316623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 39), 'idx', False)
        # Getting the type of 'Hmag' (line 477)
        Hmag_316624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 34), 'Hmag', False)
        # Obtaining the member '__getitem__' of a type (line 477)
        getitem___316625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 34), Hmag_316624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 477)
        subscript_call_result_316626 = invoke(stypy.reporting.localization.Localization(__file__, 477, 34), getitem___316625, idx_316623)
        
        int_316627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 45), 'int')
        # Processing the call keyword arguments (line 477)
        int_316628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 56), 'int')
        keyword_316629 = int_316628
        kwargs_316630 = {'decimal': keyword_316629}
        # Getting the type of 'assert_array_almost_equal' (line 477)
        assert_array_almost_equal_316622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 477)
        assert_array_almost_equal_call_result_316631 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), assert_array_almost_equal_316622, *[subscript_call_result_316626, int_316627], **kwargs_316630)
        
        
        # ################# End of 'test_firls(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_firls' in the type store
        # Getting the type of 'stypy_return_type' (line 444)
        stypy_return_type_316632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_firls'
        return stypy_return_type_316632


    @norecursion
    def test_compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_compare'
        module_type_store = module_type_store.open_function_context('test_compare', 479, 4, False)
        # Assigning a type to the variable 'self' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestFirls.test_compare.__dict__.__setitem__('stypy_localization', localization)
        TestFirls.test_compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestFirls.test_compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestFirls.test_compare.__dict__.__setitem__('stypy_function_name', 'TestFirls.test_compare')
        TestFirls.test_compare.__dict__.__setitem__('stypy_param_names_list', [])
        TestFirls.test_compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestFirls.test_compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestFirls.test_compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestFirls.test_compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestFirls.test_compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestFirls.test_compare.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirls.test_compare', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_compare', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_compare(...)' code ##################

        
        # Assigning a Call to a Name (line 481):
        
        # Assigning a Call to a Name (line 481):
        
        # Call to firls(...): (line 481)
        # Processing the call arguments (line 481)
        int_316634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 21), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 481)
        list_316635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 481)
        # Adding element type (line 481)
        int_316636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 24), list_316635, int_316636)
        # Adding element type (line 481)
        float_316637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 24), list_316635, float_316637)
        # Adding element type (line 481)
        float_316638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 24), list_316635, float_316638)
        # Adding element type (line 481)
        int_316639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 24), list_316635, int_316639)
        
        
        # Obtaining an instance of the builtin type 'list' (line 481)
        list_316640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 481)
        # Adding element type (line 481)
        int_316641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 43), list_316640, int_316641)
        # Adding element type (line 481)
        int_316642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 43), list_316640, int_316642)
        # Adding element type (line 481)
        int_316643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 43), list_316640, int_316643)
        # Adding element type (line 481)
        int_316644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 43), list_316640, int_316644)
        
        
        # Obtaining an instance of the builtin type 'list' (line 481)
        list_316645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 481)
        # Adding element type (line 481)
        int_316646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 57), list_316645, int_316646)
        # Adding element type (line 481)
        int_316647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 481, 57), list_316645, int_316647)
        
        # Processing the call keyword arguments (line 481)
        kwargs_316648 = {}
        # Getting the type of 'firls' (line 481)
        firls_316633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'firls', False)
        # Calling firls(args, kwargs) (line 481)
        firls_call_result_316649 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), firls_316633, *[int_316634, list_316635, list_316640, list_316645], **kwargs_316648)
        
        # Assigning a type to the variable 'taps' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'taps', firls_call_result_316649)
        
        # Assigning a List to a Name (line 483):
        
        # Assigning a List to a Name (line 483):
        
        # Obtaining an instance of the builtin type 'list' (line 483)
        list_316650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 483)
        # Adding element type (line 483)
        float_316651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316651)
        # Adding element type (line 483)
        float_316652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316652)
        # Adding element type (line 483)
        float_316653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316653)
        # Adding element type (line 483)
        float_316654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316654)
        # Adding element type (line 483)
        float_316655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316655)
        # Adding element type (line 483)
        float_316656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316656)
        # Adding element type (line 483)
        float_316657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316657)
        # Adding element type (line 483)
        float_316658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316658)
        # Adding element type (line 483)
        float_316659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 21), list_316650, float_316659)
        
        # Assigning a type to the variable 'known_taps' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'known_taps', list_316650)
        
        # Call to assert_allclose(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'taps' (line 488)
        taps_316661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'taps', False)
        # Getting the type of 'known_taps' (line 488)
        known_taps_316662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 30), 'known_taps', False)
        # Processing the call keyword arguments (line 488)
        kwargs_316663 = {}
        # Getting the type of 'assert_allclose' (line 488)
        assert_allclose_316660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 488)
        assert_allclose_call_result_316664 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), assert_allclose_316660, *[taps_316661, known_taps_316662], **kwargs_316663)
        
        
        # Assigning a Call to a Name (line 491):
        
        # Assigning a Call to a Name (line 491):
        
        # Call to firls(...): (line 491)
        # Processing the call arguments (line 491)
        int_316666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 21), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_316667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        int_316668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 25), list_316667, int_316668)
        # Adding element type (line 491)
        float_316669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 25), list_316667, float_316669)
        # Adding element type (line 491)
        float_316670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 25), list_316667, float_316670)
        # Adding element type (line 491)
        int_316671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 25), list_316667, int_316671)
        
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_316672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        int_316673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 43), list_316672, int_316673)
        # Adding element type (line 491)
        int_316674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 43), list_316672, int_316674)
        # Adding element type (line 491)
        int_316675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 43), list_316672, int_316675)
        # Adding element type (line 491)
        int_316676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 43), list_316672, int_316676)
        
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_316677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        int_316678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 57), list_316677, int_316678)
        # Adding element type (line 491)
        int_316679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 57), list_316677, int_316679)
        
        # Processing the call keyword arguments (line 491)
        kwargs_316680 = {}
        # Getting the type of 'firls' (line 491)
        firls_316665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 15), 'firls', False)
        # Calling firls(args, kwargs) (line 491)
        firls_call_result_316681 = invoke(stypy.reporting.localization.Localization(__file__, 491, 15), firls_316665, *[int_316666, list_316667, list_316672, list_316677], **kwargs_316680)
        
        # Assigning a type to the variable 'taps' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'taps', firls_call_result_316681)
        
        # Assigning a List to a Name (line 493):
        
        # Assigning a List to a Name (line 493):
        
        # Obtaining an instance of the builtin type 'list' (line 493)
        list_316682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 493)
        # Adding element type (line 493)
        float_316683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316683)
        # Adding element type (line 493)
        float_316684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316684)
        # Adding element type (line 493)
        float_316685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316685)
        # Adding element type (line 493)
        float_316686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316686)
        # Adding element type (line 493)
        float_316687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316687)
        # Adding element type (line 493)
        float_316688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316688)
        # Adding element type (line 493)
        float_316689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316689)
        # Adding element type (line 493)
        float_316690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316690)
        # Adding element type (line 493)
        float_316691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316691)
        # Adding element type (line 493)
        float_316692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316692)
        # Adding element type (line 493)
        float_316693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 21), list_316682, float_316693)
        
        # Assigning a type to the variable 'known_taps' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'known_taps', list_316682)
        
        # Call to assert_allclose(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'taps' (line 498)
        taps_316695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'taps', False)
        # Getting the type of 'known_taps' (line 498)
        known_taps_316696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 30), 'known_taps', False)
        # Processing the call keyword arguments (line 498)
        kwargs_316697 = {}
        # Getting the type of 'assert_allclose' (line 498)
        assert_allclose_316694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 498)
        assert_allclose_call_result_316698 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), assert_allclose_316694, *[taps_316695, known_taps_316696], **kwargs_316697)
        
        
        # Assigning a Call to a Name (line 501):
        
        # Assigning a Call to a Name (line 501):
        
        # Call to firls(...): (line 501)
        # Processing the call arguments (line 501)
        int_316700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 21), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 501)
        tuple_316701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 501)
        # Adding element type (line 501)
        int_316702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316702)
        # Adding element type (line 501)
        int_316703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316703)
        # Adding element type (line 501)
        int_316704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316704)
        # Adding element type (line 501)
        int_316705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316705)
        # Adding element type (line 501)
        int_316706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316706)
        # Adding element type (line 501)
        int_316707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 25), tuple_316701, int_316707)
        
        
        # Obtaining an instance of the builtin type 'list' (line 501)
        list_316708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 501)
        # Adding element type (line 501)
        int_316709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316709)
        # Adding element type (line 501)
        int_316710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316710)
        # Adding element type (line 501)
        int_316711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316711)
        # Adding element type (line 501)
        int_316712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316712)
        # Adding element type (line 501)
        int_316713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316713)
        # Adding element type (line 501)
        int_316714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 44), list_316708, int_316714)
        
        # Processing the call keyword arguments (line 501)
        int_316715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 67), 'int')
        keyword_316716 = int_316715
        kwargs_316717 = {'fs': keyword_316716}
        # Getting the type of 'firls' (line 501)
        firls_316699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 'firls', False)
        # Calling firls(args, kwargs) (line 501)
        firls_call_result_316718 = invoke(stypy.reporting.localization.Localization(__file__, 501, 15), firls_316699, *[int_316700, tuple_316701, list_316708], **kwargs_316717)
        
        # Assigning a type to the variable 'taps' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'taps', firls_call_result_316718)
        
        # Assigning a List to a Name (line 503):
        
        # Assigning a List to a Name (line 503):
        
        # Obtaining an instance of the builtin type 'list' (line 503)
        list_316719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 503)
        # Adding element type (line 503)
        float_316720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316720)
        # Adding element type (line 503)
        float_316721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316721)
        # Adding element type (line 503)
        float_316722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316722)
        # Adding element type (line 503)
        float_316723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316723)
        # Adding element type (line 503)
        float_316724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316724)
        # Adding element type (line 503)
        float_316725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316725)
        # Adding element type (line 503)
        float_316726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 12), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 503, 21), list_316719, float_316726)
        
        # Assigning a type to the variable 'known_taps' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'known_taps', list_316719)
        
        # Call to assert_allclose(...): (line 507)
        # Processing the call arguments (line 507)
        # Getting the type of 'taps' (line 507)
        taps_316728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'taps', False)
        # Getting the type of 'known_taps' (line 507)
        known_taps_316729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 30), 'known_taps', False)
        # Processing the call keyword arguments (line 507)
        kwargs_316730 = {}
        # Getting the type of 'assert_allclose' (line 507)
        assert_allclose_316727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 507)
        assert_allclose_call_result_316731 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), assert_allclose_316727, *[taps_316728, known_taps_316729], **kwargs_316730)
        
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to firls(...): (line 509)
        # Processing the call arguments (line 509)
        int_316733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 21), 'int')
        
        # Obtaining an instance of the builtin type 'tuple' (line 509)
        tuple_316734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 509)
        # Adding element type (line 509)
        int_316735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316735)
        # Adding element type (line 509)
        int_316736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316736)
        # Adding element type (line 509)
        int_316737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316737)
        # Adding element type (line 509)
        int_316738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316738)
        # Adding element type (line 509)
        int_316739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316739)
        # Adding element type (line 509)
        int_316740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 25), tuple_316734, int_316740)
        
        
        # Obtaining an instance of the builtin type 'list' (line 509)
        list_316741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 509)
        # Adding element type (line 509)
        int_316742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316742)
        # Adding element type (line 509)
        int_316743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316743)
        # Adding element type (line 509)
        int_316744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316744)
        # Adding element type (line 509)
        int_316745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316745)
        # Adding element type (line 509)
        int_316746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316746)
        # Adding element type (line 509)
        int_316747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 509, 44), list_316741, int_316747)
        
        # Processing the call keyword arguments (line 509)
        int_316748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 68), 'int')
        keyword_316749 = int_316748
        kwargs_316750 = {'nyq': keyword_316749}
        # Getting the type of 'firls' (line 509)
        firls_316732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 15), 'firls', False)
        # Calling firls(args, kwargs) (line 509)
        firls_call_result_316751 = invoke(stypy.reporting.localization.Localization(__file__, 509, 15), firls_316732, *[int_316733, tuple_316734, list_316741], **kwargs_316750)
        
        # Assigning a type to the variable 'taps' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'taps', firls_call_result_316751)
        
        # Call to assert_allclose(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'taps' (line 510)
        taps_316753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 24), 'taps', False)
        # Getting the type of 'known_taps' (line 510)
        known_taps_316754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 30), 'known_taps', False)
        # Processing the call keyword arguments (line 510)
        kwargs_316755 = {}
        # Getting the type of 'assert_allclose' (line 510)
        assert_allclose_316752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 510)
        assert_allclose_call_result_316756 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), assert_allclose_316752, *[taps_316753, known_taps_316754], **kwargs_316755)
        
        
        # ################# End of 'test_compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_compare' in the type store
        # Getting the type of 'stypy_return_type' (line 479)
        stypy_return_type_316757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316757)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_compare'
        return stypy_return_type_316757


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 423, 0, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestFirls.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestFirls' (line 423)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 0), 'TestFirls', TestFirls)
# Declaration of the 'TestMinimumPhase' class

class TestMinimumPhase(object, ):

    @norecursion
    def test_bad_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bad_args'
        module_type_store = module_type_store.open_function_context('test_bad_args', 515, 4, False)
        # Assigning a type to the variable 'self' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_localization', localization)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_function_name', 'TestMinimumPhase.test_bad_args')
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_param_names_list', [])
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMinimumPhase.test_bad_args.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMinimumPhase.test_bad_args', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bad_args', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bad_args(...)' code ##################

        
        # Call to assert_raises(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'ValueError' (line 517)
        ValueError_316759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 517)
        minimum_phase_316760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 34), 'minimum_phase', False)
        
        # Obtaining an instance of the builtin type 'list' (line 517)
        list_316761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 517)
        # Adding element type (line 517)
        float_316762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 49), list_316761, float_316762)
        
        # Processing the call keyword arguments (line 517)
        kwargs_316763 = {}
        # Getting the type of 'assert_raises' (line 517)
        assert_raises_316758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 517)
        assert_raises_call_result_316764 = invoke(stypy.reporting.localization.Localization(__file__, 517, 8), assert_raises_316758, *[ValueError_316759, minimum_phase_316760, list_316761], **kwargs_316763)
        
        
        # Call to assert_raises(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'ValueError' (line 518)
        ValueError_316766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 518)
        minimum_phase_316767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 34), 'minimum_phase', False)
        
        # Obtaining an instance of the builtin type 'list' (line 518)
        list_316768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 518)
        # Adding element type (line 518)
        float_316769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 49), list_316768, float_316769)
        # Adding element type (line 518)
        float_316770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 49), list_316768, float_316770)
        
        # Processing the call keyword arguments (line 518)
        kwargs_316771 = {}
        # Getting the type of 'assert_raises' (line 518)
        assert_raises_316765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 518)
        assert_raises_call_result_316772 = invoke(stypy.reporting.localization.Localization(__file__, 518, 8), assert_raises_316765, *[ValueError_316766, minimum_phase_316767, list_316768], **kwargs_316771)
        
        
        # Call to assert_raises(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'ValueError' (line 519)
        ValueError_316774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 519)
        minimum_phase_316775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 34), 'minimum_phase', False)
        
        # Call to ones(...): (line 519)
        # Processing the call arguments (line 519)
        int_316778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 57), 'int')
        # Processing the call keyword arguments (line 519)
        kwargs_316779 = {}
        # Getting the type of 'np' (line 519)
        np_316776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 49), 'np', False)
        # Obtaining the member 'ones' of a type (line 519)
        ones_316777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 49), np_316776, 'ones')
        # Calling ones(args, kwargs) (line 519)
        ones_call_result_316780 = invoke(stypy.reporting.localization.Localization(__file__, 519, 49), ones_316777, *[int_316778], **kwargs_316779)
        
        complex_316781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 63), 'complex')
        # Applying the binary operator '*' (line 519)
        result_mul_316782 = python_operator(stypy.reporting.localization.Localization(__file__, 519, 49), '*', ones_call_result_316780, complex_316781)
        
        # Processing the call keyword arguments (line 519)
        kwargs_316783 = {}
        # Getting the type of 'assert_raises' (line 519)
        assert_raises_316773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 519)
        assert_raises_call_result_316784 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), assert_raises_316773, *[ValueError_316774, minimum_phase_316775, result_mul_316782], **kwargs_316783)
        
        
        # Call to assert_raises(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 'ValueError' (line 520)
        ValueError_316786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 520)
        minimum_phase_316787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 34), 'minimum_phase', False)
        str_316788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 49), 'str', 'foo')
        # Processing the call keyword arguments (line 520)
        kwargs_316789 = {}
        # Getting the type of 'assert_raises' (line 520)
        assert_raises_316785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 520)
        assert_raises_call_result_316790 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), assert_raises_316785, *[ValueError_316786, minimum_phase_316787, str_316788], **kwargs_316789)
        
        
        # Call to assert_raises(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'ValueError' (line 521)
        ValueError_316792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 521)
        minimum_phase_316793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'minimum_phase', False)
        
        # Call to ones(...): (line 521)
        # Processing the call arguments (line 521)
        int_316796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 57), 'int')
        # Processing the call keyword arguments (line 521)
        kwargs_316797 = {}
        # Getting the type of 'np' (line 521)
        np_316794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 49), 'np', False)
        # Obtaining the member 'ones' of a type (line 521)
        ones_316795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 49), np_316794, 'ones')
        # Calling ones(args, kwargs) (line 521)
        ones_call_result_316798 = invoke(stypy.reporting.localization.Localization(__file__, 521, 49), ones_316795, *[int_316796], **kwargs_316797)
        
        # Processing the call keyword arguments (line 521)
        int_316799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 68), 'int')
        keyword_316800 = int_316799
        kwargs_316801 = {'n_fft': keyword_316800}
        # Getting the type of 'assert_raises' (line 521)
        assert_raises_316791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 521)
        assert_raises_call_result_316802 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), assert_raises_316791, *[ValueError_316792, minimum_phase_316793, ones_call_result_316798], **kwargs_316801)
        
        
        # Call to assert_raises(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'ValueError' (line 522)
        ValueError_316804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 22), 'ValueError', False)
        # Getting the type of 'minimum_phase' (line 522)
        minimum_phase_316805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 34), 'minimum_phase', False)
        
        # Call to ones(...): (line 522)
        # Processing the call arguments (line 522)
        int_316808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 57), 'int')
        # Processing the call keyword arguments (line 522)
        kwargs_316809 = {}
        # Getting the type of 'np' (line 522)
        np_316806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 49), 'np', False)
        # Obtaining the member 'ones' of a type (line 522)
        ones_316807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 49), np_316806, 'ones')
        # Calling ones(args, kwargs) (line 522)
        ones_call_result_316810 = invoke(stypy.reporting.localization.Localization(__file__, 522, 49), ones_316807, *[int_316808], **kwargs_316809)
        
        # Processing the call keyword arguments (line 522)
        str_316811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 69), 'str', 'foo')
        keyword_316812 = str_316811
        kwargs_316813 = {'method': keyword_316812}
        # Getting the type of 'assert_raises' (line 522)
        assert_raises_316803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 522)
        assert_raises_call_result_316814 = invoke(stypy.reporting.localization.Localization(__file__, 522, 8), assert_raises_316803, *[ValueError_316804, minimum_phase_316805, ones_call_result_316810], **kwargs_316813)
        
        
        # Call to assert_warns(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 'RuntimeWarning' (line 523)
        RuntimeWarning_316816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 21), 'RuntimeWarning', False)
        # Getting the type of 'minimum_phase' (line 523)
        minimum_phase_316817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 37), 'minimum_phase', False)
        
        # Call to arange(...): (line 523)
        # Processing the call arguments (line 523)
        int_316820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 62), 'int')
        # Processing the call keyword arguments (line 523)
        kwargs_316821 = {}
        # Getting the type of 'np' (line 523)
        np_316818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 52), 'np', False)
        # Obtaining the member 'arange' of a type (line 523)
        arange_316819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 52), np_316818, 'arange')
        # Calling arange(args, kwargs) (line 523)
        arange_call_result_316822 = invoke(stypy.reporting.localization.Localization(__file__, 523, 52), arange_316819, *[int_316820], **kwargs_316821)
        
        # Processing the call keyword arguments (line 523)
        kwargs_316823 = {}
        # Getting the type of 'assert_warns' (line 523)
        assert_warns_316815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'assert_warns', False)
        # Calling assert_warns(args, kwargs) (line 523)
        assert_warns_call_result_316824 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), assert_warns_316815, *[RuntimeWarning_316816, minimum_phase_316817, arange_call_result_316822], **kwargs_316823)
        
        
        # ################# End of 'test_bad_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bad_args' in the type store
        # Getting the type of 'stypy_return_type' (line 515)
        stypy_return_type_316825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bad_args'
        return stypy_return_type_316825


    @norecursion
    def test_homomorphic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_homomorphic'
        module_type_store = module_type_store.open_function_context('test_homomorphic', 525, 4, False)
        # Assigning a type to the variable 'self' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_localization', localization)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_function_name', 'TestMinimumPhase.test_homomorphic')
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_param_names_list', [])
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMinimumPhase.test_homomorphic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMinimumPhase.test_homomorphic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_homomorphic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_homomorphic(...)' code ##################

        
        # Assigning a List to a Name (line 530):
        
        # Assigning a List to a Name (line 530):
        
        # Obtaining an instance of the builtin type 'list' (line 530)
        list_316826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 530)
        # Adding element type (line 530)
        int_316827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 12), list_316826, int_316827)
        # Adding element type (line 530)
        int_316828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 530, 12), list_316826, int_316828)
        
        # Assigning a type to the variable 'h' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'h', list_316826)
        
        # Assigning a Call to a Name (line 531):
        
        # Assigning a Call to a Name (line 531):
        
        # Call to minimum_phase(...): (line 531)
        # Processing the call arguments (line 531)
        
        # Call to convolve(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 'h' (line 531)
        h_316832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 42), 'h', False)
        
        # Obtaining the type of the subscript
        int_316833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 49), 'int')
        slice_316834 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 531, 45), None, None, int_316833)
        # Getting the type of 'h' (line 531)
        h_316835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 45), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 531)
        getitem___316836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 45), h_316835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 531)
        subscript_call_result_316837 = invoke(stypy.reporting.localization.Localization(__file__, 531, 45), getitem___316836, slice_316834)
        
        # Processing the call keyword arguments (line 531)
        kwargs_316838 = {}
        # Getting the type of 'np' (line 531)
        np_316830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 30), 'np', False)
        # Obtaining the member 'convolve' of a type (line 531)
        convolve_316831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 30), np_316830, 'convolve')
        # Calling convolve(args, kwargs) (line 531)
        convolve_call_result_316839 = invoke(stypy.reporting.localization.Localization(__file__, 531, 30), convolve_316831, *[h_316832, subscript_call_result_316837], **kwargs_316838)
        
        # Processing the call keyword arguments (line 531)
        kwargs_316840 = {}
        # Getting the type of 'minimum_phase' (line 531)
        minimum_phase_316829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'minimum_phase', False)
        # Calling minimum_phase(args, kwargs) (line 531)
        minimum_phase_call_result_316841 = invoke(stypy.reporting.localization.Localization(__file__, 531, 16), minimum_phase_316829, *[convolve_call_result_316839], **kwargs_316840)
        
        # Assigning a type to the variable 'h_new' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'h_new', minimum_phase_call_result_316841)
        
        # Call to assert_allclose(...): (line 532)
        # Processing the call arguments (line 532)
        # Getting the type of 'h_new' (line 532)
        h_new_316843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 24), 'h_new', False)
        # Getting the type of 'h' (line 532)
        h_316844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 31), 'h', False)
        # Processing the call keyword arguments (line 532)
        float_316845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 39), 'float')
        keyword_316846 = float_316845
        kwargs_316847 = {'rtol': keyword_316846}
        # Getting the type of 'assert_allclose' (line 532)
        assert_allclose_316842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 532)
        assert_allclose_call_result_316848 = invoke(stypy.reporting.localization.Localization(__file__, 532, 8), assert_allclose_316842, *[h_new_316843, h_316844], **kwargs_316847)
        
        
        # Assigning a Call to a Name (line 535):
        
        # Assigning a Call to a Name (line 535):
        
        # Call to RandomState(...): (line 535)
        # Processing the call arguments (line 535)
        int_316852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 36), 'int')
        # Processing the call keyword arguments (line 535)
        kwargs_316853 = {}
        # Getting the type of 'np' (line 535)
        np_316849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 14), 'np', False)
        # Obtaining the member 'random' of a type (line 535)
        random_316850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 14), np_316849, 'random')
        # Obtaining the member 'RandomState' of a type (line 535)
        RandomState_316851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 14), random_316850, 'RandomState')
        # Calling RandomState(args, kwargs) (line 535)
        RandomState_call_result_316854 = invoke(stypy.reporting.localization.Localization(__file__, 535, 14), RandomState_316851, *[int_316852], **kwargs_316853)
        
        # Assigning a type to the variable 'rng' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'rng', RandomState_call_result_316854)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 536)
        tuple_316855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 536)
        # Adding element type (line 536)
        int_316856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316856)
        # Adding element type (line 536)
        int_316857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316857)
        # Adding element type (line 536)
        int_316858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316858)
        # Adding element type (line 536)
        int_316859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316859)
        # Adding element type (line 536)
        int_316860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316860)
        # Adding element type (line 536)
        int_316861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316861)
        # Adding element type (line 536)
        int_316862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316862)
        # Adding element type (line 536)
        int_316863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316863)
        # Adding element type (line 536)
        int_316864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316864)
        # Adding element type (line 536)
        int_316865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316865)
        # Adding element type (line 536)
        int_316866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 18), tuple_316855, int_316866)
        
        # Testing the type of a for loop iterable (line 536)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 536, 8), tuple_316855)
        # Getting the type of the for loop variable (line 536)
        for_loop_var_316867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 536, 8), tuple_316855)
        # Assigning a type to the variable 'n' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'n', for_loop_var_316867)
        # SSA begins for a for statement (line 536)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 537):
        
        # Assigning a Call to a Name (line 537):
        
        # Call to randn(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'n' (line 537)
        n_316870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 26), 'n', False)
        # Processing the call keyword arguments (line 537)
        kwargs_316871 = {}
        # Getting the type of 'rng' (line 537)
        rng_316868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 16), 'rng', False)
        # Obtaining the member 'randn' of a type (line 537)
        randn_316869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 16), rng_316868, 'randn')
        # Calling randn(args, kwargs) (line 537)
        randn_call_result_316872 = invoke(stypy.reporting.localization.Localization(__file__, 537, 16), randn_316869, *[n_316870], **kwargs_316871)
        
        # Assigning a type to the variable 'h' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'h', randn_call_result_316872)
        
        # Assigning a Call to a Name (line 538):
        
        # Assigning a Call to a Name (line 538):
        
        # Call to minimum_phase(...): (line 538)
        # Processing the call arguments (line 538)
        
        # Call to convolve(...): (line 538)
        # Processing the call arguments (line 538)
        # Getting the type of 'h' (line 538)
        h_316876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 46), 'h', False)
        
        # Obtaining the type of the subscript
        int_316877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 53), 'int')
        slice_316878 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 538, 49), None, None, int_316877)
        # Getting the type of 'h' (line 538)
        h_316879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 49), 'h', False)
        # Obtaining the member '__getitem__' of a type (line 538)
        getitem___316880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 49), h_316879, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 538)
        subscript_call_result_316881 = invoke(stypy.reporting.localization.Localization(__file__, 538, 49), getitem___316880, slice_316878)
        
        # Processing the call keyword arguments (line 538)
        kwargs_316882 = {}
        # Getting the type of 'np' (line 538)
        np_316874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 34), 'np', False)
        # Obtaining the member 'convolve' of a type (line 538)
        convolve_316875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 34), np_316874, 'convolve')
        # Calling convolve(args, kwargs) (line 538)
        convolve_call_result_316883 = invoke(stypy.reporting.localization.Localization(__file__, 538, 34), convolve_316875, *[h_316876, subscript_call_result_316881], **kwargs_316882)
        
        # Processing the call keyword arguments (line 538)
        kwargs_316884 = {}
        # Getting the type of 'minimum_phase' (line 538)
        minimum_phase_316873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 20), 'minimum_phase', False)
        # Calling minimum_phase(args, kwargs) (line 538)
        minimum_phase_call_result_316885 = invoke(stypy.reporting.localization.Localization(__file__, 538, 20), minimum_phase_316873, *[convolve_call_result_316883], **kwargs_316884)
        
        # Assigning a type to the variable 'h_new' (line 538)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 12), 'h_new', minimum_phase_call_result_316885)
        
        # Call to assert_allclose(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to abs(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to fft(...): (line 539)
        # Processing the call arguments (line 539)
        # Getting the type of 'h_new' (line 539)
        h_new_316892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 46), 'h_new', False)
        # Processing the call keyword arguments (line 539)
        kwargs_316893 = {}
        # Getting the type of 'np' (line 539)
        np_316889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 35), 'np', False)
        # Obtaining the member 'fft' of a type (line 539)
        fft_316890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 35), np_316889, 'fft')
        # Obtaining the member 'fft' of a type (line 539)
        fft_316891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 35), fft_316890, 'fft')
        # Calling fft(args, kwargs) (line 539)
        fft_call_result_316894 = invoke(stypy.reporting.localization.Localization(__file__, 539, 35), fft_316891, *[h_new_316892], **kwargs_316893)
        
        # Processing the call keyword arguments (line 539)
        kwargs_316895 = {}
        # Getting the type of 'np' (line 539)
        np_316887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 28), 'np', False)
        # Obtaining the member 'abs' of a type (line 539)
        abs_316888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 28), np_316887, 'abs')
        # Calling abs(args, kwargs) (line 539)
        abs_call_result_316896 = invoke(stypy.reporting.localization.Localization(__file__, 539, 28), abs_316888, *[fft_call_result_316894], **kwargs_316895)
        
        
        # Call to abs(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Call to fft(...): (line 540)
        # Processing the call arguments (line 540)
        # Getting the type of 'h' (line 540)
        h_316902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 46), 'h', False)
        # Processing the call keyword arguments (line 540)
        kwargs_316903 = {}
        # Getting the type of 'np' (line 540)
        np_316899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 35), 'np', False)
        # Obtaining the member 'fft' of a type (line 540)
        fft_316900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 35), np_316899, 'fft')
        # Obtaining the member 'fft' of a type (line 540)
        fft_316901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 35), fft_316900, 'fft')
        # Calling fft(args, kwargs) (line 540)
        fft_call_result_316904 = invoke(stypy.reporting.localization.Localization(__file__, 540, 35), fft_316901, *[h_316902], **kwargs_316903)
        
        # Processing the call keyword arguments (line 540)
        kwargs_316905 = {}
        # Getting the type of 'np' (line 540)
        np_316897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 28), 'np', False)
        # Obtaining the member 'abs' of a type (line 540)
        abs_316898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 28), np_316897, 'abs')
        # Calling abs(args, kwargs) (line 540)
        abs_call_result_316906 = invoke(stypy.reporting.localization.Localization(__file__, 540, 28), abs_316898, *[fft_call_result_316904], **kwargs_316905)
        
        # Processing the call keyword arguments (line 539)
        float_316907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 56), 'float')
        keyword_316908 = float_316907
        kwargs_316909 = {'rtol': keyword_316908}
        # Getting the type of 'assert_allclose' (line 539)
        assert_allclose_316886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 539)
        assert_allclose_call_result_316910 = invoke(stypy.reporting.localization.Localization(__file__, 539, 12), assert_allclose_316886, *[abs_call_result_316896, abs_call_result_316906], **kwargs_316909)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_homomorphic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_homomorphic' in the type store
        # Getting the type of 'stypy_return_type' (line 525)
        stypy_return_type_316911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_homomorphic'
        return stypy_return_type_316911


    @norecursion
    def test_hilbert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hilbert'
        module_type_store = module_type_store.open_function_context('test_hilbert', 542, 4, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_localization', localization)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_function_name', 'TestMinimumPhase.test_hilbert')
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_param_names_list', [])
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestMinimumPhase.test_hilbert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMinimumPhase.test_hilbert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hilbert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hilbert(...)' code ##################

        
        # Assigning a Call to a Name (line 548):
        
        # Assigning a Call to a Name (line 548):
        
        # Call to remez(...): (line 548)
        # Processing the call arguments (line 548)
        int_316913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 548)
        list_316914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 548)
        # Adding element type (line 548)
        int_316915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 22), list_316914, int_316915)
        # Adding element type (line 548)
        float_316916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 22), list_316914, float_316916)
        # Adding element type (line 548)
        float_316917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 22), list_316914, float_316917)
        # Adding element type (line 548)
        int_316918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 22), list_316914, int_316918)
        
        
        # Obtaining an instance of the builtin type 'list' (line 548)
        list_316919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 548)
        # Adding element type (line 548)
        int_316920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 40), list_316919, int_316920)
        # Adding element type (line 548)
        int_316921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 548, 40), list_316919, int_316921)
        
        # Processing the call keyword arguments (line 548)
        float_316922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 51), 'float')
        keyword_316923 = float_316922
        kwargs_316924 = {'fs': keyword_316923}
        # Getting the type of 'remez' (line 548)
        remez_316912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'remez', False)
        # Calling remez(args, kwargs) (line 548)
        remez_call_result_316925 = invoke(stypy.reporting.localization.Localization(__file__, 548, 12), remez_316912, *[int_316913, list_316914, list_316919], **kwargs_316924)
        
        # Assigning a type to the variable 'h' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'h', remez_call_result_316925)
        
        # Assigning a List to a Name (line 549):
        
        # Assigning a List to a Name (line 549):
        
        # Obtaining an instance of the builtin type 'list' (line 549)
        list_316926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 549)
        # Adding element type (line 549)
        float_316927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316927)
        # Adding element type (line 549)
        float_316928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316928)
        # Adding element type (line 549)
        float_316929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316929)
        # Adding element type (line 549)
        float_316930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316930)
        # Adding element type (line 549)
        float_316931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316931)
        # Adding element type (line 549)
        float_316932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 549, 12), list_316926, float_316932)
        
        # Assigning a type to the variable 'k' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'k', list_316926)
        
        # Assigning a Call to a Name (line 551):
        
        # Assigning a Call to a Name (line 551):
        
        # Call to minimum_phase(...): (line 551)
        # Processing the call arguments (line 551)
        # Getting the type of 'h' (line 551)
        h_316934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 26), 'h', False)
        str_316935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 29), 'str', 'hilbert')
        # Processing the call keyword arguments (line 551)
        kwargs_316936 = {}
        # Getting the type of 'minimum_phase' (line 551)
        minimum_phase_316933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'minimum_phase', False)
        # Calling minimum_phase(args, kwargs) (line 551)
        minimum_phase_call_result_316937 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), minimum_phase_316933, *[h_316934, str_316935], **kwargs_316936)
        
        # Assigning a type to the variable 'm' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'm', minimum_phase_call_result_316937)
        
        # Call to assert_allclose(...): (line 552)
        # Processing the call arguments (line 552)
        # Getting the type of 'm' (line 552)
        m_316939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 24), 'm', False)
        # Getting the type of 'k' (line 552)
        k_316940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 27), 'k', False)
        # Processing the call keyword arguments (line 552)
        float_316941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 35), 'float')
        keyword_316942 = float_316941
        kwargs_316943 = {'rtol': keyword_316942}
        # Getting the type of 'assert_allclose' (line 552)
        assert_allclose_316938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 552)
        assert_allclose_call_result_316944 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), assert_allclose_316938, *[m_316939, k_316940], **kwargs_316943)
        
        
        # Assigning a Call to a Name (line 557):
        
        # Assigning a Call to a Name (line 557):
        
        # Call to remez(...): (line 557)
        # Processing the call arguments (line 557)
        int_316946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 18), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 557)
        list_316947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 557)
        # Adding element type (line 557)
        int_316948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 22), list_316947, int_316948)
        # Adding element type (line 557)
        float_316949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 22), list_316947, float_316949)
        # Adding element type (line 557)
        float_316950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 22), list_316947, float_316950)
        # Adding element type (line 557)
        int_316951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 22), list_316947, int_316951)
        
        
        # Obtaining an instance of the builtin type 'list' (line 557)
        list_316952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 557)
        # Adding element type (line 557)
        int_316953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 40), list_316952, int_316953)
        # Adding element type (line 557)
        int_316954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 40), list_316952, int_316954)
        
        # Processing the call keyword arguments (line 557)
        float_316955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 51), 'float')
        keyword_316956 = float_316955
        kwargs_316957 = {'fs': keyword_316956}
        # Getting the type of 'remez' (line 557)
        remez_316945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'remez', False)
        # Calling remez(args, kwargs) (line 557)
        remez_call_result_316958 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), remez_316945, *[int_316946, list_316947, list_316952], **kwargs_316957)
        
        # Assigning a type to the variable 'h' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'h', remez_call_result_316958)
        
        # Assigning a List to a Name (line 558):
        
        # Assigning a List to a Name (line 558):
        
        # Obtaining an instance of the builtin type 'list' (line 558)
        list_316959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 558)
        # Adding element type (line 558)
        float_316960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316960)
        # Adding element type (line 558)
        float_316961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316961)
        # Adding element type (line 558)
        float_316962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316962)
        # Adding element type (line 558)
        float_316963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316963)
        # Adding element type (line 558)
        float_316964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316964)
        # Adding element type (line 558)
        float_316965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316965)
        # Adding element type (line 558)
        float_316966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316966)
        # Adding element type (line 558)
        float_316967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316967)
        # Adding element type (line 558)
        float_316968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316968)
        # Adding element type (line 558)
        float_316969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316969)
        # Adding element type (line 558)
        float_316970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 12), list_316959, float_316970)
        
        # Assigning a type to the variable 'k' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'k', list_316959)
        
        # Assigning a Call to a Name (line 562):
        
        # Assigning a Call to a Name (line 562):
        
        # Call to minimum_phase(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'h' (line 562)
        h_316972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 26), 'h', False)
        str_316973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 29), 'str', 'hilbert')
        # Processing the call keyword arguments (line 562)
        int_316974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 46), 'int')
        int_316975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 49), 'int')
        # Applying the binary operator '**' (line 562)
        result_pow_316976 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 46), '**', int_316974, int_316975)
        
        keyword_316977 = result_pow_316976
        kwargs_316978 = {'n_fft': keyword_316977}
        # Getting the type of 'minimum_phase' (line 562)
        minimum_phase_316971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 12), 'minimum_phase', False)
        # Calling minimum_phase(args, kwargs) (line 562)
        minimum_phase_call_result_316979 = invoke(stypy.reporting.localization.Localization(__file__, 562, 12), minimum_phase_316971, *[h_316972, str_316973], **kwargs_316978)
        
        # Assigning a type to the variable 'm' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'm', minimum_phase_call_result_316979)
        
        # Call to assert_allclose(...): (line 563)
        # Processing the call arguments (line 563)
        # Getting the type of 'm' (line 563)
        m_316981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 24), 'm', False)
        # Getting the type of 'k' (line 563)
        k_316982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 27), 'k', False)
        # Processing the call keyword arguments (line 563)
        float_316983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 35), 'float')
        keyword_316984 = float_316983
        kwargs_316985 = {'rtol': keyword_316984}
        # Getting the type of 'assert_allclose' (line 563)
        assert_allclose_316980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 563)
        assert_allclose_call_result_316986 = invoke(stypy.reporting.localization.Localization(__file__, 563, 8), assert_allclose_316980, *[m_316981, k_316982], **kwargs_316985)
        
        
        # ################# End of 'test_hilbert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hilbert' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_316987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_316987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hilbert'
        return stypy_return_type_316987


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 513, 0, False)
        # Assigning a type to the variable 'self' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestMinimumPhase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestMinimumPhase' (line 513)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 0), 'TestMinimumPhase', TestMinimumPhase)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
