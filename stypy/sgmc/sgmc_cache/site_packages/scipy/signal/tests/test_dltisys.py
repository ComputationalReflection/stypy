
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Author: Jeffrey Armstrong <jeff@approximatrix.com>
2: # April 4, 2011
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import numpy as np
7: from numpy.testing import (assert_equal,
8:                            assert_array_almost_equal, assert_array_equal,
9:                            assert_allclose, assert_, assert_almost_equal)
10: from pytest import raises as assert_raises
11: from scipy._lib._numpy_compat import suppress_warnings
12: from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
13:                           StateSpace, TransferFunction, ZerosPolesGain,
14:                           dfreqresp, dbode, BadCoefficients)
15: 
16: 
17: class TestDLTI(object):
18: 
19:     def test_dlsim(self):
20: 
21:         a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
22:         b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
23:         c = np.asarray([[0.1, 0.3]])
24:         d = np.asarray([[0.0, -0.1, 0.0]])
25:         dt = 0.5
26: 
27:         # Create an input matrix with inputs down the columns (3 cols) and its
28:         # respective time input vector
29:         u = np.hstack((np.asmatrix(np.linspace(0, 4.0, num=5)).transpose(),
30:                        0.01 * np.ones((5, 1)),
31:                        -0.002 * np.ones((5, 1))))
32:         t_in = np.linspace(0, 2.0, num=5)
33: 
34:         # Define the known result
35:         yout_truth = np.asmatrix([-0.001,
36:                                   -0.00073,
37:                                   0.039446,
38:                                   0.0915387,
39:                                   0.13195948]).transpose()
40:         xout_truth = np.asarray([[0, 0],
41:                                  [0.0012, 0.0005],
42:                                  [0.40233, 0.00071],
43:                                  [1.163368, -0.079327],
44:                                  [2.2402985, -0.3035679]])
45: 
46:         tout, yout, xout = dlsim((a, b, c, d, dt), u, t_in)
47: 
48:         assert_array_almost_equal(yout_truth, yout)
49:         assert_array_almost_equal(xout_truth, xout)
50:         assert_array_almost_equal(t_in, tout)
51: 
52:         # Make sure input with single-dimension doesn't raise error
53:         dlsim((1, 2, 3), 4)
54: 
55:         # Interpolated control - inputs should have different time steps
56:         # than the discrete model uses internally
57:         u_sparse = u[[0, 4], :]
58:         t_sparse = np.asarray([0.0, 2.0])
59: 
60:         tout, yout, xout = dlsim((a, b, c, d, dt), u_sparse, t_sparse)
61: 
62:         assert_array_almost_equal(yout_truth, yout)
63:         assert_array_almost_equal(xout_truth, xout)
64:         assert_equal(len(tout), yout.shape[0])
65: 
66:         # Transfer functions (assume dt = 0.5)
67:         num = np.asarray([1.0, -0.1])
68:         den = np.asarray([0.3, 1.0, 0.2])
69:         yout_truth = np.asmatrix([0.0,
70:                                   0.0,
71:                                   3.33333333333333,
72:                                   -4.77777777777778,
73:                                   23.0370370370370]).transpose()
74: 
75:         # Assume use of the first column of the control input built earlier
76:         tout, yout = dlsim((num, den, 0.5), u[:, 0], t_in)
77: 
78:         assert_array_almost_equal(yout, yout_truth)
79:         assert_array_almost_equal(t_in, tout)
80: 
81:         # Retest the same with a 1-D input vector
82:         uflat = np.asarray(u[:, 0])
83:         uflat = uflat.reshape((5,))
84:         tout, yout = dlsim((num, den, 0.5), uflat, t_in)
85: 
86:         assert_array_almost_equal(yout, yout_truth)
87:         assert_array_almost_equal(t_in, tout)
88: 
89:         # zeros-poles-gain representation
90:         zd = np.array([0.5, -0.5])
91:         pd = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
92:         k = 1.0
93:         yout_truth = np.asmatrix([0.0, 1.0, 2.0, 2.25, 2.5]).transpose()
94: 
95:         tout, yout = dlsim((zd, pd, k, 0.5), u[:, 0], t_in)
96: 
97:         assert_array_almost_equal(yout, yout_truth)
98:         assert_array_almost_equal(t_in, tout)
99: 
100:         # Raise an error for continuous-time systems
101:         system = lti([1], [1, 1])
102:         assert_raises(AttributeError, dlsim, system, u)
103: 
104:     def test_dstep(self):
105: 
106:         a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
107:         b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
108:         c = np.asarray([[0.1, 0.3]])
109:         d = np.asarray([[0.0, -0.1, 0.0]])
110:         dt = 0.5
111: 
112:         # Because b.shape[1] == 3, dstep should result in a tuple of three
113:         # result vectors
114:         yout_step_truth = (np.asarray([0.0, 0.04, 0.052, 0.0404, 0.00956,
115:                                        -0.036324, -0.093318, -0.15782348,
116:                                        -0.226628324, -0.2969374948]),
117:                            np.asarray([-0.1, -0.075, -0.058, -0.04815,
118:                                        -0.04453, -0.0461895, -0.0521812,
119:                                        -0.061588875, -0.073549579,
120:                                        -0.08727047595]),
121:                            np.asarray([0.0, -0.01, -0.013, -0.0101, -0.00239,
122:                                        0.009081, 0.0233295, 0.03945587,
123:                                        0.056657081, 0.0742343737]))
124: 
125:         tout, yout = dstep((a, b, c, d, dt), n=10)
126: 
127:         assert_equal(len(yout), 3)
128: 
129:         for i in range(0, len(yout)):
130:             assert_equal(yout[i].shape[0], 10)
131:             assert_array_almost_equal(yout[i].flatten(), yout_step_truth[i])
132: 
133:         # Check that the other two inputs (tf, zpk) will work as well
134:         tfin = ([1.0], [1.0, 1.0], 0.5)
135:         yout_tfstep = np.asarray([0.0, 1.0, 0.0])
136:         tout, yout = dstep(tfin, n=3)
137:         assert_equal(len(yout), 1)
138:         assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
139: 
140:         zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
141:         tout, yout = dstep(zpkin, n=3)
142:         assert_equal(len(yout), 1)
143:         assert_array_almost_equal(yout[0].flatten(), yout_tfstep)
144: 
145:         # Raise an error for continuous-time systems
146:         system = lti([1], [1, 1])
147:         assert_raises(AttributeError, dstep, system)
148: 
149:     def test_dimpulse(self):
150: 
151:         a = np.asarray([[0.9, 0.1], [-0.2, 0.9]])
152:         b = np.asarray([[0.4, 0.1, -0.1], [0.0, 0.05, 0.0]])
153:         c = np.asarray([[0.1, 0.3]])
154:         d = np.asarray([[0.0, -0.1, 0.0]])
155:         dt = 0.5
156: 
157:         # Because b.shape[1] == 3, dimpulse should result in a tuple of three
158:         # result vectors
159:         yout_imp_truth = (np.asarray([0.0, 0.04, 0.012, -0.0116, -0.03084,
160:                                       -0.045884, -0.056994, -0.06450548,
161:                                       -0.068804844, -0.0703091708]),
162:                           np.asarray([-0.1, 0.025, 0.017, 0.00985, 0.00362,
163:                                       -0.0016595, -0.0059917, -0.009407675,
164:                                       -0.011960704, -0.01372089695]),
165:                           np.asarray([0.0, -0.01, -0.003, 0.0029, 0.00771,
166:                                       0.011471, 0.0142485, 0.01612637,
167:                                       0.017201211, 0.0175772927]))
168: 
169:         tout, yout = dimpulse((a, b, c, d, dt), n=10)
170: 
171:         assert_equal(len(yout), 3)
172: 
173:         for i in range(0, len(yout)):
174:             assert_equal(yout[i].shape[0], 10)
175:             assert_array_almost_equal(yout[i].flatten(), yout_imp_truth[i])
176: 
177:         # Check that the other two inputs (tf, zpk) will work as well
178:         tfin = ([1.0], [1.0, 1.0], 0.5)
179:         yout_tfimpulse = np.asarray([0.0, 1.0, -1.0])
180:         tout, yout = dimpulse(tfin, n=3)
181:         assert_equal(len(yout), 1)
182:         assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
183: 
184:         zpkin = tf2zpk(tfin[0], tfin[1]) + (0.5,)
185:         tout, yout = dimpulse(zpkin, n=3)
186:         assert_equal(len(yout), 1)
187:         assert_array_almost_equal(yout[0].flatten(), yout_tfimpulse)
188: 
189:         # Raise an error for continuous-time systems
190:         system = lti([1], [1, 1])
191:         assert_raises(AttributeError, dimpulse, system)
192: 
193:     def test_dlsim_trivial(self):
194:         a = np.array([[0.0]])
195:         b = np.array([[0.0]])
196:         c = np.array([[0.0]])
197:         d = np.array([[0.0]])
198:         n = 5
199:         u = np.zeros(n).reshape(-1, 1)
200:         tout, yout, xout = dlsim((a, b, c, d, 1), u)
201:         assert_array_equal(tout, np.arange(float(n)))
202:         assert_array_equal(yout, np.zeros((n, 1)))
203:         assert_array_equal(xout, np.zeros((n, 1)))
204: 
205:     def test_dlsim_simple1d(self):
206:         a = np.array([[0.5]])
207:         b = np.array([[0.0]])
208:         c = np.array([[1.0]])
209:         d = np.array([[0.0]])
210:         n = 5
211:         u = np.zeros(n).reshape(-1, 1)
212:         tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
213:         assert_array_equal(tout, np.arange(float(n)))
214:         expected = (0.5 ** np.arange(float(n))).reshape(-1, 1)
215:         assert_array_equal(yout, expected)
216:         assert_array_equal(xout, expected)
217: 
218:     def test_dlsim_simple2d(self):
219:         lambda1 = 0.5
220:         lambda2 = 0.25
221:         a = np.array([[lambda1, 0.0],
222:                       [0.0, lambda2]])
223:         b = np.array([[0.0],
224:                       [0.0]])
225:         c = np.array([[1.0, 0.0],
226:                       [0.0, 1.0]])
227:         d = np.array([[0.0],
228:                       [0.0]])
229:         n = 5
230:         u = np.zeros(n).reshape(-1, 1)
231:         tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
232:         assert_array_equal(tout, np.arange(float(n)))
233:         # The analytical solution:
234:         expected = (np.array([lambda1, lambda2]) **
235:                                 np.arange(float(n)).reshape(-1, 1))
236:         assert_array_equal(yout, expected)
237:         assert_array_equal(xout, expected)
238: 
239:     def test_more_step_and_impulse(self):
240:         lambda1 = 0.5
241:         lambda2 = 0.75
242:         a = np.array([[lambda1, 0.0],
243:                       [0.0, lambda2]])
244:         b = np.array([[1.0, 0.0],
245:                       [0.0, 1.0]])
246:         c = np.array([[1.0, 1.0]])
247:         d = np.array([[0.0, 0.0]])
248: 
249:         n = 10
250: 
251:         # Check a step response.
252:         ts, ys = dstep((a, b, c, d, 1), n=n)
253: 
254:         # Create the exact step response.
255:         stp0 = (1.0 / (1 - lambda1)) * (1.0 - lambda1 ** np.arange(n))
256:         stp1 = (1.0 / (1 - lambda2)) * (1.0 - lambda2 ** np.arange(n))
257: 
258:         assert_allclose(ys[0][:, 0], stp0)
259:         assert_allclose(ys[1][:, 0], stp1)
260: 
261:         # Check an impulse response with an initial condition.
262:         x0 = np.array([1.0, 1.0])
263:         ti, yi = dimpulse((a, b, c, d, 1), n=n, x0=x0)
264: 
265:         # Create the exact impulse response.
266:         imp = (np.array([lambda1, lambda2]) **
267:                             np.arange(-1, n + 1).reshape(-1, 1))
268:         imp[0, :] = 0.0
269:         # Analytical solution to impulse response
270:         y0 = imp[:n, 0] + np.dot(imp[1:n + 1, :], x0)
271:         y1 = imp[:n, 1] + np.dot(imp[1:n + 1, :], x0)
272: 
273:         assert_allclose(yi[0][:, 0], y0)
274:         assert_allclose(yi[1][:, 0], y1)
275: 
276:         # Check that dt=0.1, n=3 gives 3 time values.
277:         system = ([1.0], [1.0, -0.5], 0.1)
278:         t, (y,) = dstep(system, n=3)
279:         assert_allclose(t, [0, 0.1, 0.2])
280:         assert_array_equal(y.T, [[0, 1.0, 1.5]])
281:         t, (y,) = dimpulse(system, n=3)
282:         assert_allclose(t, [0, 0.1, 0.2])
283:         assert_array_equal(y.T, [[0, 1, 0.5]])
284: 
285: 
286: class TestDlti(object):
287:     def test_dlti_instantiation(self):
288:         # Test that lti can be instantiated.
289: 
290:         dt = 0.05
291:         # TransferFunction
292:         s = dlti([1], [-1], dt=dt)
293:         assert_(isinstance(s, TransferFunction))
294:         assert_(isinstance(s, dlti))
295:         assert_(not isinstance(s, lti))
296:         assert_equal(s.dt, dt)
297: 
298:         # ZerosPolesGain
299:         s = dlti(np.array([]), np.array([-1]), 1, dt=dt)
300:         assert_(isinstance(s, ZerosPolesGain))
301:         assert_(isinstance(s, dlti))
302:         assert_(not isinstance(s, lti))
303:         assert_equal(s.dt, dt)
304: 
305:         # StateSpace
306:         s = dlti([1], [-1], 1, 3, dt=dt)
307:         assert_(isinstance(s, StateSpace))
308:         assert_(isinstance(s, dlti))
309:         assert_(not isinstance(s, lti))
310:         assert_equal(s.dt, dt)
311: 
312:         # Number of inputs
313:         assert_raises(ValueError, dlti, 1)
314:         assert_raises(ValueError, dlti, 1, 1, 1, 1, 1)
315: 
316: 
317: class TestStateSpaceDisc(object):
318:     def test_initialization(self):
319:         # Check that all initializations work
320:         dt = 0.05
321:         s = StateSpace(1, 1, 1, 1, dt=dt)
322:         s = StateSpace([1], [2], [3], [4], dt=dt)
323:         s = StateSpace(np.array([[1, 2], [3, 4]]), np.array([[1], [2]]),
324:                        np.array([[1, 0]]), np.array([[0]]), dt=dt)
325:         s = StateSpace(1, 1, 1, 1, dt=True)
326: 
327:     def test_conversion(self):
328:         # Check the conversion functions
329:         s = StateSpace(1, 2, 3, 4, dt=0.05)
330:         assert_(isinstance(s.to_ss(), StateSpace))
331:         assert_(isinstance(s.to_tf(), TransferFunction))
332:         assert_(isinstance(s.to_zpk(), ZerosPolesGain))
333: 
334:         # Make sure copies work
335:         assert_(StateSpace(s) is not s)
336:         assert_(s.to_ss() is not s)
337: 
338:     def test_properties(self):
339:         # Test setters/getters for cross class properties.
340:         # This implicitly tests to_tf() and to_zpk()
341: 
342:         # Getters
343:         s = StateSpace(1, 1, 1, 1, dt=0.05)
344:         assert_equal(s.poles, [1])
345:         assert_equal(s.zeros, [0])
346: 
347: 
348: class TestTransferFunction(object):
349:     def test_initialization(self):
350:         # Check that all initializations work
351:         dt = 0.05
352:         s = TransferFunction(1, 1, dt=dt)
353:         s = TransferFunction([1], [2], dt=dt)
354:         s = TransferFunction(np.array([1]), np.array([2]), dt=dt)
355:         s = TransferFunction(1, 1, dt=True)
356: 
357:     def test_conversion(self):
358:         # Check the conversion functions
359:         s = TransferFunction([1, 0], [1, -1], dt=0.05)
360:         assert_(isinstance(s.to_ss(), StateSpace))
361:         assert_(isinstance(s.to_tf(), TransferFunction))
362:         assert_(isinstance(s.to_zpk(), ZerosPolesGain))
363: 
364:         # Make sure copies work
365:         assert_(TransferFunction(s) is not s)
366:         assert_(s.to_tf() is not s)
367: 
368:     def test_properties(self):
369:         # Test setters/getters for cross class properties.
370:         # This implicitly tests to_ss() and to_zpk()
371: 
372:         # Getters
373:         s = TransferFunction([1, 0], [1, -1], dt=0.05)
374:         assert_equal(s.poles, [1])
375:         assert_equal(s.zeros, [0])
376: 
377: 
378: class TestZerosPolesGain(object):
379:     def test_initialization(self):
380:         # Check that all initializations work
381:         dt = 0.05
382:         s = ZerosPolesGain(1, 1, 1, dt=dt)
383:         s = ZerosPolesGain([1], [2], 1, dt=dt)
384:         s = ZerosPolesGain(np.array([1]), np.array([2]), 1, dt=dt)
385:         s = ZerosPolesGain(1, 1, 1, dt=True)
386: 
387:     def test_conversion(self):
388:         # Check the conversion functions
389:         s = ZerosPolesGain(1, 2, 3, dt=0.05)
390:         assert_(isinstance(s.to_ss(), StateSpace))
391:         assert_(isinstance(s.to_tf(), TransferFunction))
392:         assert_(isinstance(s.to_zpk(), ZerosPolesGain))
393: 
394:         # Make sure copies work
395:         assert_(ZerosPolesGain(s) is not s)
396:         assert_(s.to_zpk() is not s)
397: 
398: 
399: class Test_dfreqresp(object):
400: 
401:     def test_manual(self):
402:         # Test dfreqresp() real part calculation (manual sanity check).
403:         # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
404:         system = TransferFunction(1, [1, -0.2], dt=0.1)
405:         w = [0.1, 1, 10]
406:         w, H = dfreqresp(system, w=w)
407: 
408:         # test real
409:         expected_re = [1.2383, 0.4130, -0.7553]
410:         assert_almost_equal(H.real, expected_re, decimal=4)
411: 
412:         # test imag
413:         expected_im = [-0.1555, -1.0214, 0.3955]
414:         assert_almost_equal(H.imag, expected_im, decimal=4)
415: 
416:     def test_auto(self):
417:         # Test dfreqresp() real part calculation.
418:         # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
419:         system = TransferFunction(1, [1, -0.2], dt=0.1)
420:         w = [0.1, 1, 10, 100]
421:         w, H = dfreqresp(system, w=w)
422:         jw = np.exp(w * 1j)
423:         y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
424: 
425:         # test real
426:         expected_re = y.real
427:         assert_almost_equal(H.real, expected_re)
428: 
429:         # test imag
430:         expected_im = y.imag
431:         assert_almost_equal(H.imag, expected_im)
432: 
433:     def test_freq_range(self):
434:         # Test that freqresp() finds a reasonable frequency range.
435:         # 1st order low-pass filter: H(z) = 1 / (z - 0.2),
436:         # Expected range is from 0.01 to 10.
437:         system = TransferFunction(1, [1, -0.2], dt=0.1)
438:         n = 10
439:         expected_w = np.linspace(0, np.pi, 10, endpoint=False)
440:         w, H = dfreqresp(system, n=n)
441:         assert_almost_equal(w, expected_w)
442: 
443:     def test_pole_one(self):
444:         # Test that freqresp() doesn't fail on a system with a pole at 0.
445:         # integrator, pole at zero: H(s) = 1 / s
446:         system = TransferFunction([1], [1, -1], dt=0.1)
447: 
448:         with suppress_warnings() as sup:
449:             sup.filter(RuntimeWarning, message="divide by zero")
450:             sup.filter(RuntimeWarning, message="invalid value encountered")
451:             w, H = dfreqresp(system, n=2)
452:         assert_equal(w[0], 0.)  # a fail would give not-a-number
453: 
454:     def test_error(self):
455:         # Raise an error for continuous-time systems
456:         system = lti([1], [1, 1])
457:         assert_raises(AttributeError, dfreqresp, system)
458: 
459:     def test_from_state_space(self):
460:         # H(z) = 2 / z^3 - 0.5 * z^2
461: 
462:         system_TF = dlti([2], [1, -0.5, 0, 0])
463: 
464:         A = np.array([[0.5, 0, 0],
465:                       [1, 0, 0],
466:                       [0, 1, 0]])
467:         B = np.array([[1, 0, 0]]).T
468:         C = np.array([[0, 0, 2]])
469:         D = 0
470: 
471:         system_SS = dlti(A, B, C, D)
472:         w = 10.0**np.arange(-3,0,.5)
473:         with suppress_warnings() as sup:
474:             sup.filter(BadCoefficients)
475:             w1, H1 = dfreqresp(system_TF, w=w)
476:             w2, H2 = dfreqresp(system_SS, w=w)
477: 
478:         assert_almost_equal(H1, H2)
479: 
480:     def test_from_zpk(self):
481:         # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
482:         system_ZPK = dlti([],[0.2],0.3)
483:         system_TF = dlti(0.3, [1, -0.2])
484:         w = [0.1, 1, 10, 100]
485:         w1, H1 = dfreqresp(system_ZPK, w=w)
486:         w2, H2 = dfreqresp(system_TF, w=w)
487:         assert_almost_equal(H1, H2)
488: 
489: 
490: class Test_bode(object):
491: 
492:     def test_manual(self):
493:         # Test bode() magnitude calculation (manual sanity check).
494:         # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
495:         dt = 0.1
496:         system = TransferFunction(0.3, [1, -0.2], dt=dt)
497:         w = [0.1, 0.5, 1, np.pi]
498:         w2, mag, phase = dbode(system, w=w)
499: 
500:         # Test mag
501:         expected_mag = [-8.5329, -8.8396, -9.6162, -12.0412]
502:         assert_almost_equal(mag, expected_mag, decimal=4)
503: 
504:         # Test phase
505:         expected_phase = [-7.1575, -35.2814, -67.9809, -180.0000]
506:         assert_almost_equal(phase, expected_phase, decimal=4)
507: 
508:         # Test frequency
509:         assert_equal(np.array(w) / dt, w2)
510: 
511:     def test_auto(self):
512:         # Test bode() magnitude calculation.
513:         # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
514:         system = TransferFunction(0.3, [1, -0.2], dt=0.1)
515:         w = np.array([0.1, 0.5, 1, np.pi])
516:         w2, mag, phase = dbode(system, w=w)
517:         jw = np.exp(w * 1j)
518:         y = np.polyval(system.num, jw) / np.polyval(system.den, jw)
519: 
520:         # Test mag
521:         expected_mag = 20.0 * np.log10(abs(y))
522:         assert_almost_equal(mag, expected_mag)
523: 
524:         # Test phase
525:         expected_phase = np.rad2deg(np.angle(y))
526:         assert_almost_equal(phase, expected_phase)
527: 
528:     def test_range(self):
529:         # Test that bode() finds a reasonable frequency range.
530:         # 1st order low-pass filter: H(s) = 0.3 / (z - 0.2),
531:         dt = 0.1
532:         system = TransferFunction(0.3, [1, -0.2], dt=0.1)
533:         n = 10
534:         # Expected range is from 0.01 to 10.
535:         expected_w = np.linspace(0, np.pi, n, endpoint=False) / dt
536:         w, mag, phase = dbode(system, n=n)
537:         assert_almost_equal(w, expected_w)
538: 
539:     def test_pole_one(self):
540:         # Test that freqresp() doesn't fail on a system with a pole at 0.
541:         # integrator, pole at zero: H(s) = 1 / s
542:         system = TransferFunction([1], [1, -1], dt=0.1)
543: 
544:         with suppress_warnings() as sup:
545:             sup.filter(RuntimeWarning, message="divide by zero")
546:             sup.filter(RuntimeWarning, message="invalid value encountered")
547:             w, mag, phase = dbode(system, n=2)
548:         assert_equal(w[0], 0.)  # a fail would give not-a-number
549: 
550:     def test_imaginary(self):
551:         # bode() should not fail on a system with pure imaginary poles.
552:         # The test passes if bode doesn't raise an exception.
553:         system = TransferFunction([1], [1, 0, 100], dt=0.1)
554:         dbode(system, n=2)
555: 
556:     def test_error(self):
557:         # Raise an error for continuous-time systems
558:         system = lti([1], [1, 1])
559:         assert_raises(AttributeError, dbode, system)
560: 
561: 
562: class TestTransferFunctionZConversion(object):
563:     '''Test private conversions between 'z' and 'z**-1' polynomials.'''
564: 
565:     def test_full(self):
566:         # Numerator and denominator same order
567:         num = [2, 3, 4]
568:         den = [5, 6, 7]
569:         num2, den2 = TransferFunction._z_to_zinv(num, den)
570:         assert_equal(num, num2)
571:         assert_equal(den, den2)
572: 
573:         num2, den2 = TransferFunction._zinv_to_z(num, den)
574:         assert_equal(num, num2)
575:         assert_equal(den, den2)
576: 
577:     def test_numerator(self):
578:         # Numerator lower order than denominator
579:         num = [2, 3]
580:         den = [5, 6, 7]
581:         num2, den2 = TransferFunction._z_to_zinv(num, den)
582:         assert_equal([0, 2, 3], num2)
583:         assert_equal(den, den2)
584: 
585:         num2, den2 = TransferFunction._zinv_to_z(num, den)
586:         assert_equal([2, 3, 0], num2)
587:         assert_equal(den, den2)
588: 
589:     def test_denominator(self):
590:         # Numerator higher order than denominator
591:         num = [2, 3, 4]
592:         den = [5, 6]
593:         num2, den2 = TransferFunction._z_to_zinv(num, den)
594:         assert_equal(num, num2)
595:         assert_equal([0, 5, 6], den2)
596: 
597:         num2, den2 = TransferFunction._zinv_to_z(num, den)
598:         assert_equal(num, num2)
599:         assert_equal([5, 6, 0], den2)
600: 
601: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_292254 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_292254) is not StypyTypeError):

    if (import_292254 != 'pyd_module'):
        __import__(import_292254)
        sys_modules_292255 = sys.modules[import_292254]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_292255.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_292254)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_equal, assert_allclose, assert_, assert_almost_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_292256 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_292256) is not StypyTypeError):

    if (import_292256 != 'pyd_module'):
        __import__(import_292256)
        sys_modules_292257 = sys.modules[import_292256]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_292257.module_type_store, module_type_store, ['assert_equal', 'assert_array_almost_equal', 'assert_array_equal', 'assert_allclose', 'assert_', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_292257, sys_modules_292257.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_equal, assert_allclose, assert_, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_almost_equal', 'assert_array_equal', 'assert_allclose', 'assert_', 'assert_almost_equal'], [assert_equal, assert_array_almost_equal, assert_array_equal, assert_allclose, assert_, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_292256)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_292258 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_292258) is not StypyTypeError):

    if (import_292258 != 'pyd_module'):
        __import__(import_292258)
        sys_modules_292259 = sys.modules[import_292258]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_292259.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_292259, sys_modules_292259.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_292258)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_292260 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_292260) is not StypyTypeError):

    if (import_292260 != 'pyd_module'):
        __import__(import_292260)
        sys_modules_292261 = sys.modules[import_292260]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_292261.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_292261, sys_modules_292261.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_292260)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.signal import dlsim, dstep, dimpulse, tf2zpk, lti, dlti, StateSpace, TransferFunction, ZerosPolesGain, dfreqresp, dbode, BadCoefficients' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_292262 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.signal')

if (type(import_292262) is not StypyTypeError):

    if (import_292262 != 'pyd_module'):
        __import__(import_292262)
        sys_modules_292263 = sys.modules[import_292262]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.signal', sys_modules_292263.module_type_store, module_type_store, ['dlsim', 'dstep', 'dimpulse', 'tf2zpk', 'lti', 'dlti', 'StateSpace', 'TransferFunction', 'ZerosPolesGain', 'dfreqresp', 'dbode', 'BadCoefficients'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_292263, sys_modules_292263.module_type_store, module_type_store)
    else:
        from scipy.signal import dlsim, dstep, dimpulse, tf2zpk, lti, dlti, StateSpace, TransferFunction, ZerosPolesGain, dfreqresp, dbode, BadCoefficients

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.signal', None, module_type_store, ['dlsim', 'dstep', 'dimpulse', 'tf2zpk', 'lti', 'dlti', 'StateSpace', 'TransferFunction', 'ZerosPolesGain', 'dfreqresp', 'dbode', 'BadCoefficients'], [dlsim, dstep, dimpulse, tf2zpk, lti, dlti, StateSpace, TransferFunction, ZerosPolesGain, dfreqresp, dbode, BadCoefficients])

else:
    # Assigning a type to the variable 'scipy.signal' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.signal', import_292262)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# Declaration of the 'TestDLTI' class

class TestDLTI(object, ):

    @norecursion
    def test_dlsim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlsim'
        module_type_store = module_type_store.open_function_context('test_dlsim', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dlsim')
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dlsim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dlsim', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlsim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlsim(...)' code ##################

        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to asarray(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_292266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_292267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        float_292268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), list_292267, float_292268)
        # Adding element type (line 21)
        float_292269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), list_292267, float_292269)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_292266, list_292267)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_292270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        float_292271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 36), list_292270, float_292271)
        # Adding element type (line 21)
        float_292272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 36), list_292270, float_292272)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_292266, list_292270)
        
        # Processing the call keyword arguments (line 21)
        kwargs_292273 = {}
        # Getting the type of 'np' (line 21)
        np_292264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 21)
        asarray_292265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), np_292264, 'asarray')
        # Calling asarray(args, kwargs) (line 21)
        asarray_call_result_292274 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), asarray_292265, *[list_292266], **kwargs_292273)
        
        # Assigning a type to the variable 'a' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'a', asarray_call_result_292274)
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to asarray(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_292277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_292278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        float_292279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 24), list_292278, float_292279)
        # Adding element type (line 22)
        float_292280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 24), list_292278, float_292280)
        # Adding element type (line 22)
        float_292281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 24), list_292278, float_292281)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_292277, list_292278)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_292282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        float_292283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_292282, float_292283)
        # Adding element type (line 22)
        float_292284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_292282, float_292284)
        # Adding element type (line 22)
        float_292285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 42), list_292282, float_292285)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 23), list_292277, list_292282)
        
        # Processing the call keyword arguments (line 22)
        kwargs_292286 = {}
        # Getting the type of 'np' (line 22)
        np_292275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 22)
        asarray_292276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), np_292275, 'asarray')
        # Calling asarray(args, kwargs) (line 22)
        asarray_call_result_292287 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), asarray_292276, *[list_292277], **kwargs_292286)
        
        # Assigning a type to the variable 'b' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'b', asarray_call_result_292287)
        
        # Assigning a Call to a Name (line 23):
        
        # Assigning a Call to a Name (line 23):
        
        # Assigning a Call to a Name (line 23):
        
        # Call to asarray(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_292290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_292291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        float_292292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), list_292291, float_292292)
        # Adding element type (line 23)
        float_292293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), list_292291, float_292293)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 23), list_292290, list_292291)
        
        # Processing the call keyword arguments (line 23)
        kwargs_292294 = {}
        # Getting the type of 'np' (line 23)
        np_292288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 23)
        asarray_292289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), np_292288, 'asarray')
        # Calling asarray(args, kwargs) (line 23)
        asarray_call_result_292295 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), asarray_292289, *[list_292290], **kwargs_292294)
        
        # Assigning a type to the variable 'c' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'c', asarray_call_result_292295)
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to asarray(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_292298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_292299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        float_292300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 24), list_292299, float_292300)
        # Adding element type (line 24)
        float_292301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 24), list_292299, float_292301)
        # Adding element type (line 24)
        float_292302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 24), list_292299, float_292302)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), list_292298, list_292299)
        
        # Processing the call keyword arguments (line 24)
        kwargs_292303 = {}
        # Getting the type of 'np' (line 24)
        np_292296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 24)
        asarray_292297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), np_292296, 'asarray')
        # Calling asarray(args, kwargs) (line 24)
        asarray_call_result_292304 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), asarray_292297, *[list_292298], **kwargs_292303)
        
        # Assigning a type to the variable 'd' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'd', asarray_call_result_292304)
        
        # Assigning a Num to a Name (line 25):
        
        # Assigning a Num to a Name (line 25):
        
        # Assigning a Num to a Name (line 25):
        float_292305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'float')
        # Assigning a type to the variable 'dt' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'dt', float_292305)
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Assigning a Call to a Name (line 29):
        
        # Call to hstack(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_292308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        
        # Call to transpose(...): (line 29)
        # Processing the call keyword arguments (line 29)
        kwargs_292322 = {}
        
        # Call to asmatrix(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to linspace(...): (line 29)
        # Processing the call arguments (line 29)
        int_292313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 47), 'int')
        float_292314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 50), 'float')
        # Processing the call keyword arguments (line 29)
        int_292315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 59), 'int')
        keyword_292316 = int_292315
        kwargs_292317 = {'num': keyword_292316}
        # Getting the type of 'np' (line 29)
        np_292311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'np', False)
        # Obtaining the member 'linspace' of a type (line 29)
        linspace_292312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 35), np_292311, 'linspace')
        # Calling linspace(args, kwargs) (line 29)
        linspace_call_result_292318 = invoke(stypy.reporting.localization.Localization(__file__, 29, 35), linspace_292312, *[int_292313, float_292314], **kwargs_292317)
        
        # Processing the call keyword arguments (line 29)
        kwargs_292319 = {}
        # Getting the type of 'np' (line 29)
        np_292309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 29)
        asmatrix_292310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), np_292309, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 29)
        asmatrix_call_result_292320 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), asmatrix_292310, *[linspace_call_result_292318], **kwargs_292319)
        
        # Obtaining the member 'transpose' of a type (line 29)
        transpose_292321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), asmatrix_call_result_292320, 'transpose')
        # Calling transpose(args, kwargs) (line 29)
        transpose_call_result_292323 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), transpose_292321, *[], **kwargs_292322)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), tuple_292308, transpose_call_result_292323)
        # Adding element type (line 29)
        float_292324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'float')
        
        # Call to ones(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_292327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        int_292328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 39), tuple_292327, int_292328)
        # Adding element type (line 30)
        int_292329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 39), tuple_292327, int_292329)
        
        # Processing the call keyword arguments (line 30)
        kwargs_292330 = {}
        # Getting the type of 'np' (line 30)
        np_292325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'np', False)
        # Obtaining the member 'ones' of a type (line 30)
        ones_292326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 30), np_292325, 'ones')
        # Calling ones(args, kwargs) (line 30)
        ones_call_result_292331 = invoke(stypy.reporting.localization.Localization(__file__, 30, 30), ones_292326, *[tuple_292327], **kwargs_292330)
        
        # Applying the binary operator '*' (line 30)
        result_mul_292332 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 23), '*', float_292324, ones_call_result_292331)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), tuple_292308, result_mul_292332)
        # Adding element type (line 29)
        float_292333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'float')
        
        # Call to ones(...): (line 31)
        # Processing the call arguments (line 31)
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_292336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        int_292337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 41), tuple_292336, int_292337)
        # Adding element type (line 31)
        int_292338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 41), tuple_292336, int_292338)
        
        # Processing the call keyword arguments (line 31)
        kwargs_292339 = {}
        # Getting the type of 'np' (line 31)
        np_292334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 32), 'np', False)
        # Obtaining the member 'ones' of a type (line 31)
        ones_292335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 32), np_292334, 'ones')
        # Calling ones(args, kwargs) (line 31)
        ones_call_result_292340 = invoke(stypy.reporting.localization.Localization(__file__, 31, 32), ones_292335, *[tuple_292336], **kwargs_292339)
        
        # Applying the binary operator '*' (line 31)
        result_mul_292341 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 23), '*', float_292333, ones_call_result_292340)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 23), tuple_292308, result_mul_292341)
        
        # Processing the call keyword arguments (line 29)
        kwargs_292342 = {}
        # Getting the type of 'np' (line 29)
        np_292306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'np', False)
        # Obtaining the member 'hstack' of a type (line 29)
        hstack_292307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), np_292306, 'hstack')
        # Calling hstack(args, kwargs) (line 29)
        hstack_call_result_292343 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), hstack_292307, *[tuple_292308], **kwargs_292342)
        
        # Assigning a type to the variable 'u' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'u', hstack_call_result_292343)
        
        # Assigning a Call to a Name (line 32):
        
        # Assigning a Call to a Name (line 32):
        
        # Assigning a Call to a Name (line 32):
        
        # Call to linspace(...): (line 32)
        # Processing the call arguments (line 32)
        int_292346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
        float_292347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 30), 'float')
        # Processing the call keyword arguments (line 32)
        int_292348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 39), 'int')
        keyword_292349 = int_292348
        kwargs_292350 = {'num': keyword_292349}
        # Getting the type of 'np' (line 32)
        np_292344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'np', False)
        # Obtaining the member 'linspace' of a type (line 32)
        linspace_292345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), np_292344, 'linspace')
        # Calling linspace(args, kwargs) (line 32)
        linspace_call_result_292351 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), linspace_292345, *[int_292346, float_292347], **kwargs_292350)
        
        # Assigning a type to the variable 't_in' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 't_in', linspace_call_result_292351)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to transpose(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_292363 = {}
        
        # Call to asmatrix(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_292354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        float_292355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), list_292354, float_292355)
        # Adding element type (line 35)
        float_292356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), list_292354, float_292356)
        # Adding element type (line 35)
        float_292357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), list_292354, float_292357)
        # Adding element type (line 35)
        float_292358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), list_292354, float_292358)
        # Adding element type (line 35)
        float_292359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), list_292354, float_292359)
        
        # Processing the call keyword arguments (line 35)
        kwargs_292360 = {}
        # Getting the type of 'np' (line 35)
        np_292352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 35)
        asmatrix_292353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), np_292352, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 35)
        asmatrix_call_result_292361 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), asmatrix_292353, *[list_292354], **kwargs_292360)
        
        # Obtaining the member 'transpose' of a type (line 35)
        transpose_292362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 21), asmatrix_call_result_292361, 'transpose')
        # Calling transpose(args, kwargs) (line 35)
        transpose_call_result_292364 = invoke(stypy.reporting.localization.Localization(__file__, 35, 21), transpose_292362, *[], **kwargs_292363)
        
        # Assigning a type to the variable 'yout_truth' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'yout_truth', transpose_call_result_292364)
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to asarray(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_292367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_292368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        int_292369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), list_292368, int_292369)
        # Adding element type (line 40)
        int_292370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 33), list_292368, int_292370)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 32), list_292367, list_292368)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_292371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        float_292372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 33), list_292371, float_292372)
        # Adding element type (line 41)
        float_292373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 33), list_292371, float_292373)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 32), list_292367, list_292371)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_292374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        float_292375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_292374, float_292375)
        # Adding element type (line 42)
        float_292376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 33), list_292374, float_292376)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 32), list_292367, list_292374)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_292377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        float_292378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 33), list_292377, float_292378)
        # Adding element type (line 43)
        float_292379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 33), list_292377, float_292379)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 32), list_292367, list_292377)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_292380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        # Adding element type (line 44)
        float_292381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 33), list_292380, float_292381)
        # Adding element type (line 44)
        float_292382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 33), list_292380, float_292382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 32), list_292367, list_292380)
        
        # Processing the call keyword arguments (line 40)
        kwargs_292383 = {}
        # Getting the type of 'np' (line 40)
        np_292365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'np', False)
        # Obtaining the member 'asarray' of a type (line 40)
        asarray_292366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 21), np_292365, 'asarray')
        # Calling asarray(args, kwargs) (line 40)
        asarray_call_result_292384 = invoke(stypy.reporting.localization.Localization(__file__, 40, 21), asarray_292366, *[list_292367], **kwargs_292383)
        
        # Assigning a type to the variable 'xout_truth' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'xout_truth', asarray_call_result_292384)
        
        # Assigning a Call to a Tuple (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_292385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to dlsim(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_292387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'a' (line 46)
        a_292388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292387, a_292388)
        # Adding element type (line 46)
        # Getting the type of 'b' (line 46)
        b_292389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292387, b_292389)
        # Adding element type (line 46)
        # Getting the type of 'c' (line 46)
        c_292390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292387, c_292390)
        # Adding element type (line 46)
        # Getting the type of 'd' (line 46)
        d_292391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292387, d_292391)
        # Adding element type (line 46)
        # Getting the type of 'dt' (line 46)
        dt_292392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292387, dt_292392)
        
        # Getting the type of 'u' (line 46)
        u_292393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 51), 'u', False)
        # Getting the type of 't_in' (line 46)
        t_in_292394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 't_in', False)
        # Processing the call keyword arguments (line 46)
        kwargs_292395 = {}
        # Getting the type of 'dlsim' (line 46)
        dlsim_292386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 46)
        dlsim_call_result_292396 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), dlsim_292386, *[tuple_292387, u_292393, t_in_292394], **kwargs_292395)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___292397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), dlsim_call_result_292396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_292398 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___292397, int_292385)
        
        # Assigning a type to the variable 'tuple_var_assignment_292171' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292171', subscript_call_result_292398)
        
        # Assigning a Subscript to a Name (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_292399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to dlsim(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_292401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'a' (line 46)
        a_292402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292401, a_292402)
        # Adding element type (line 46)
        # Getting the type of 'b' (line 46)
        b_292403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292401, b_292403)
        # Adding element type (line 46)
        # Getting the type of 'c' (line 46)
        c_292404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292401, c_292404)
        # Adding element type (line 46)
        # Getting the type of 'd' (line 46)
        d_292405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292401, d_292405)
        # Adding element type (line 46)
        # Getting the type of 'dt' (line 46)
        dt_292406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292401, dt_292406)
        
        # Getting the type of 'u' (line 46)
        u_292407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 51), 'u', False)
        # Getting the type of 't_in' (line 46)
        t_in_292408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 't_in', False)
        # Processing the call keyword arguments (line 46)
        kwargs_292409 = {}
        # Getting the type of 'dlsim' (line 46)
        dlsim_292400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 46)
        dlsim_call_result_292410 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), dlsim_292400, *[tuple_292401, u_292407, t_in_292408], **kwargs_292409)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___292411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), dlsim_call_result_292410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_292412 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___292411, int_292399)
        
        # Assigning a type to the variable 'tuple_var_assignment_292172' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292172', subscript_call_result_292412)
        
        # Assigning a Subscript to a Name (line 46):
        
        # Assigning a Subscript to a Name (line 46):
        
        # Obtaining the type of the subscript
        int_292413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'int')
        
        # Call to dlsim(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_292415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        # Getting the type of 'a' (line 46)
        a_292416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292415, a_292416)
        # Adding element type (line 46)
        # Getting the type of 'b' (line 46)
        b_292417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292415, b_292417)
        # Adding element type (line 46)
        # Getting the type of 'c' (line 46)
        c_292418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292415, c_292418)
        # Adding element type (line 46)
        # Getting the type of 'd' (line 46)
        d_292419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292415, d_292419)
        # Adding element type (line 46)
        # Getting the type of 'dt' (line 46)
        dt_292420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_292415, dt_292420)
        
        # Getting the type of 'u' (line 46)
        u_292421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 51), 'u', False)
        # Getting the type of 't_in' (line 46)
        t_in_292422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 't_in', False)
        # Processing the call keyword arguments (line 46)
        kwargs_292423 = {}
        # Getting the type of 'dlsim' (line 46)
        dlsim_292414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 46)
        dlsim_call_result_292424 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), dlsim_292414, *[tuple_292415, u_292421, t_in_292422], **kwargs_292423)
        
        # Obtaining the member '__getitem__' of a type (line 46)
        getitem___292425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), dlsim_call_result_292424, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 46)
        subscript_call_result_292426 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), getitem___292425, int_292413)
        
        # Assigning a type to the variable 'tuple_var_assignment_292173' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292173', subscript_call_result_292426)
        
        # Assigning a Name to a Name (line 46):
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_292171' (line 46)
        tuple_var_assignment_292171_292427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292171')
        # Assigning a type to the variable 'tout' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tout', tuple_var_assignment_292171_292427)
        
        # Assigning a Name to a Name (line 46):
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_292172' (line 46)
        tuple_var_assignment_292172_292428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292172')
        # Assigning a type to the variable 'yout' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'yout', tuple_var_assignment_292172_292428)
        
        # Assigning a Name to a Name (line 46):
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'tuple_var_assignment_292173' (line 46)
        tuple_var_assignment_292173_292429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'tuple_var_assignment_292173')
        # Assigning a type to the variable 'xout' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'xout', tuple_var_assignment_292173_292429)
        
        # Call to assert_array_almost_equal(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'yout_truth' (line 48)
        yout_truth_292431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'yout_truth', False)
        # Getting the type of 'yout' (line 48)
        yout_292432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 46), 'yout', False)
        # Processing the call keyword arguments (line 48)
        kwargs_292433 = {}
        # Getting the type of 'assert_array_almost_equal' (line 48)
        assert_array_almost_equal_292430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 48)
        assert_array_almost_equal_call_result_292434 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_array_almost_equal_292430, *[yout_truth_292431, yout_292432], **kwargs_292433)
        
        
        # Call to assert_array_almost_equal(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'xout_truth' (line 49)
        xout_truth_292436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'xout_truth', False)
        # Getting the type of 'xout' (line 49)
        xout_292437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 46), 'xout', False)
        # Processing the call keyword arguments (line 49)
        kwargs_292438 = {}
        # Getting the type of 'assert_array_almost_equal' (line 49)
        assert_array_almost_equal_292435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 49)
        assert_array_almost_equal_call_result_292439 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_array_almost_equal_292435, *[xout_truth_292436, xout_292437], **kwargs_292438)
        
        
        # Call to assert_array_almost_equal(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 't_in' (line 50)
        t_in_292441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 't_in', False)
        # Getting the type of 'tout' (line 50)
        tout_292442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'tout', False)
        # Processing the call keyword arguments (line 50)
        kwargs_292443 = {}
        # Getting the type of 'assert_array_almost_equal' (line 50)
        assert_array_almost_equal_292440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 50)
        assert_array_almost_equal_call_result_292444 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert_array_almost_equal_292440, *[t_in_292441, tout_292442], **kwargs_292443)
        
        
        # Call to dlsim(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_292446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        int_292447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), tuple_292446, int_292447)
        # Adding element type (line 53)
        int_292448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), tuple_292446, int_292448)
        # Adding element type (line 53)
        int_292449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), tuple_292446, int_292449)
        
        int_292450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'int')
        # Processing the call keyword arguments (line 53)
        kwargs_292451 = {}
        # Getting the type of 'dlsim' (line 53)
        dlsim_292445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 53)
        dlsim_call_result_292452 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), dlsim_292445, *[tuple_292446, int_292450], **kwargs_292451)
        
        
        # Assigning a Subscript to a Name (line 57):
        
        # Assigning a Subscript to a Name (line 57):
        
        # Assigning a Subscript to a Name (line 57):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_292453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        int_292454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_292453, int_292454)
        # Adding element type (line 57)
        int_292455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_292453, int_292455)
        
        slice_292456 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 19), None, None, None)
        # Getting the type of 'u' (line 57)
        u_292457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'u')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___292458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), u_292457, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_292459 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), getitem___292458, (list_292453, slice_292456))
        
        # Assigning a type to the variable 'u_sparse' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'u_sparse', subscript_call_result_292459)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to asarray(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_292462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        float_292463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), list_292462, float_292463)
        # Adding element type (line 58)
        float_292464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 30), list_292462, float_292464)
        
        # Processing the call keyword arguments (line 58)
        kwargs_292465 = {}
        # Getting the type of 'np' (line 58)
        np_292460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 58)
        asarray_292461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), np_292460, 'asarray')
        # Calling asarray(args, kwargs) (line 58)
        asarray_call_result_292466 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), asarray_292461, *[list_292462], **kwargs_292465)
        
        # Assigning a type to the variable 't_sparse' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 't_sparse', asarray_call_result_292466)
        
        # Assigning a Call to a Tuple (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_292467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to dlsim(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_292469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        # Getting the type of 'a' (line 60)
        a_292470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292469, a_292470)
        # Adding element type (line 60)
        # Getting the type of 'b' (line 60)
        b_292471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292469, b_292471)
        # Adding element type (line 60)
        # Getting the type of 'c' (line 60)
        c_292472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292469, c_292472)
        # Adding element type (line 60)
        # Getting the type of 'd' (line 60)
        d_292473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292469, d_292473)
        # Adding element type (line 60)
        # Getting the type of 'dt' (line 60)
        dt_292474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292469, dt_292474)
        
        # Getting the type of 'u_sparse' (line 60)
        u_sparse_292475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'u_sparse', False)
        # Getting the type of 't_sparse' (line 60)
        t_sparse_292476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 't_sparse', False)
        # Processing the call keyword arguments (line 60)
        kwargs_292477 = {}
        # Getting the type of 'dlsim' (line 60)
        dlsim_292468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 60)
        dlsim_call_result_292478 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), dlsim_292468, *[tuple_292469, u_sparse_292475, t_sparse_292476], **kwargs_292477)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___292479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), dlsim_call_result_292478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_292480 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___292479, int_292467)
        
        # Assigning a type to the variable 'tuple_var_assignment_292174' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292174', subscript_call_result_292480)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_292481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to dlsim(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_292483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        # Getting the type of 'a' (line 60)
        a_292484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292483, a_292484)
        # Adding element type (line 60)
        # Getting the type of 'b' (line 60)
        b_292485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292483, b_292485)
        # Adding element type (line 60)
        # Getting the type of 'c' (line 60)
        c_292486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292483, c_292486)
        # Adding element type (line 60)
        # Getting the type of 'd' (line 60)
        d_292487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292483, d_292487)
        # Adding element type (line 60)
        # Getting the type of 'dt' (line 60)
        dt_292488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292483, dt_292488)
        
        # Getting the type of 'u_sparse' (line 60)
        u_sparse_292489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'u_sparse', False)
        # Getting the type of 't_sparse' (line 60)
        t_sparse_292490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 't_sparse', False)
        # Processing the call keyword arguments (line 60)
        kwargs_292491 = {}
        # Getting the type of 'dlsim' (line 60)
        dlsim_292482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 60)
        dlsim_call_result_292492 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), dlsim_292482, *[tuple_292483, u_sparse_292489, t_sparse_292490], **kwargs_292491)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___292493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), dlsim_call_result_292492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_292494 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___292493, int_292481)
        
        # Assigning a type to the variable 'tuple_var_assignment_292175' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292175', subscript_call_result_292494)
        
        # Assigning a Subscript to a Name (line 60):
        
        # Assigning a Subscript to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_292495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'int')
        
        # Call to dlsim(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_292497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        # Getting the type of 'a' (line 60)
        a_292498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292497, a_292498)
        # Adding element type (line 60)
        # Getting the type of 'b' (line 60)
        b_292499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292497, b_292499)
        # Adding element type (line 60)
        # Getting the type of 'c' (line 60)
        c_292500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292497, c_292500)
        # Adding element type (line 60)
        # Getting the type of 'd' (line 60)
        d_292501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292497, d_292501)
        # Adding element type (line 60)
        # Getting the type of 'dt' (line 60)
        dt_292502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 46), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 34), tuple_292497, dt_292502)
        
        # Getting the type of 'u_sparse' (line 60)
        u_sparse_292503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'u_sparse', False)
        # Getting the type of 't_sparse' (line 60)
        t_sparse_292504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 61), 't_sparse', False)
        # Processing the call keyword arguments (line 60)
        kwargs_292505 = {}
        # Getting the type of 'dlsim' (line 60)
        dlsim_292496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 60)
        dlsim_call_result_292506 = invoke(stypy.reporting.localization.Localization(__file__, 60, 27), dlsim_292496, *[tuple_292497, u_sparse_292503, t_sparse_292504], **kwargs_292505)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___292507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), dlsim_call_result_292506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_292508 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), getitem___292507, int_292495)
        
        # Assigning a type to the variable 'tuple_var_assignment_292176' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292176', subscript_call_result_292508)
        
        # Assigning a Name to a Name (line 60):
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_292174' (line 60)
        tuple_var_assignment_292174_292509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292174')
        # Assigning a type to the variable 'tout' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tout', tuple_var_assignment_292174_292509)
        
        # Assigning a Name to a Name (line 60):
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_292175' (line 60)
        tuple_var_assignment_292175_292510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292175')
        # Assigning a type to the variable 'yout' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'yout', tuple_var_assignment_292175_292510)
        
        # Assigning a Name to a Name (line 60):
        
        # Assigning a Name to a Name (line 60):
        # Getting the type of 'tuple_var_assignment_292176' (line 60)
        tuple_var_assignment_292176_292511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'tuple_var_assignment_292176')
        # Assigning a type to the variable 'xout' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'xout', tuple_var_assignment_292176_292511)
        
        # Call to assert_array_almost_equal(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'yout_truth' (line 62)
        yout_truth_292513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 'yout_truth', False)
        # Getting the type of 'yout' (line 62)
        yout_292514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 46), 'yout', False)
        # Processing the call keyword arguments (line 62)
        kwargs_292515 = {}
        # Getting the type of 'assert_array_almost_equal' (line 62)
        assert_array_almost_equal_292512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 62)
        assert_array_almost_equal_call_result_292516 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), assert_array_almost_equal_292512, *[yout_truth_292513, yout_292514], **kwargs_292515)
        
        
        # Call to assert_array_almost_equal(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'xout_truth' (line 63)
        xout_truth_292518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 34), 'xout_truth', False)
        # Getting the type of 'xout' (line 63)
        xout_292519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 46), 'xout', False)
        # Processing the call keyword arguments (line 63)
        kwargs_292520 = {}
        # Getting the type of 'assert_array_almost_equal' (line 63)
        assert_array_almost_equal_292517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 63)
        assert_array_almost_equal_call_result_292521 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), assert_array_almost_equal_292517, *[xout_truth_292518, xout_292519], **kwargs_292520)
        
        
        # Call to assert_equal(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to len(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'tout' (line 64)
        tout_292524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'tout', False)
        # Processing the call keyword arguments (line 64)
        kwargs_292525 = {}
        # Getting the type of 'len' (line 64)
        len_292523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'len', False)
        # Calling len(args, kwargs) (line 64)
        len_call_result_292526 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), len_292523, *[tout_292524], **kwargs_292525)
        
        
        # Obtaining the type of the subscript
        int_292527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'int')
        # Getting the type of 'yout' (line 64)
        yout_292528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'yout', False)
        # Obtaining the member 'shape' of a type (line 64)
        shape_292529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), yout_292528, 'shape')
        # Obtaining the member '__getitem__' of a type (line 64)
        getitem___292530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), shape_292529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 64)
        subscript_call_result_292531 = invoke(stypy.reporting.localization.Localization(__file__, 64, 32), getitem___292530, int_292527)
        
        # Processing the call keyword arguments (line 64)
        kwargs_292532 = {}
        # Getting the type of 'assert_equal' (line 64)
        assert_equal_292522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 64)
        assert_equal_call_result_292533 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_equal_292522, *[len_call_result_292526, subscript_call_result_292531], **kwargs_292532)
        
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to asarray(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_292536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        float_292537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), list_292536, float_292537)
        # Adding element type (line 67)
        float_292538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 25), list_292536, float_292538)
        
        # Processing the call keyword arguments (line 67)
        kwargs_292539 = {}
        # Getting the type of 'np' (line 67)
        np_292534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 67)
        asarray_292535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 14), np_292534, 'asarray')
        # Calling asarray(args, kwargs) (line 67)
        asarray_call_result_292540 = invoke(stypy.reporting.localization.Localization(__file__, 67, 14), asarray_292535, *[list_292536], **kwargs_292539)
        
        # Assigning a type to the variable 'num' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'num', asarray_call_result_292540)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to asarray(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_292543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        float_292544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_292543, float_292544)
        # Adding element type (line 68)
        float_292545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_292543, float_292545)
        # Adding element type (line 68)
        float_292546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 25), list_292543, float_292546)
        
        # Processing the call keyword arguments (line 68)
        kwargs_292547 = {}
        # Getting the type of 'np' (line 68)
        np_292541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'np', False)
        # Obtaining the member 'asarray' of a type (line 68)
        asarray_292542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), np_292541, 'asarray')
        # Calling asarray(args, kwargs) (line 68)
        asarray_call_result_292548 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), asarray_292542, *[list_292543], **kwargs_292547)
        
        # Assigning a type to the variable 'den' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'den', asarray_call_result_292548)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to transpose(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_292560 = {}
        
        # Call to asmatrix(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_292551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        float_292552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), list_292551, float_292552)
        # Adding element type (line 69)
        float_292553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), list_292551, float_292553)
        # Adding element type (line 69)
        float_292554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), list_292551, float_292554)
        # Adding element type (line 69)
        float_292555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), list_292551, float_292555)
        # Adding element type (line 69)
        float_292556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 33), list_292551, float_292556)
        
        # Processing the call keyword arguments (line 69)
        kwargs_292557 = {}
        # Getting the type of 'np' (line 69)
        np_292549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 69)
        asmatrix_292550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), np_292549, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 69)
        asmatrix_call_result_292558 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), asmatrix_292550, *[list_292551], **kwargs_292557)
        
        # Obtaining the member 'transpose' of a type (line 69)
        transpose_292559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 21), asmatrix_call_result_292558, 'transpose')
        # Calling transpose(args, kwargs) (line 69)
        transpose_call_result_292561 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), transpose_292559, *[], **kwargs_292560)
        
        # Assigning a type to the variable 'yout_truth' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'yout_truth', transpose_call_result_292561)
        
        # Assigning a Call to a Tuple (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_292562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        
        # Call to dlsim(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_292564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'num' (line 76)
        num_292565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292564, num_292565)
        # Adding element type (line 76)
        # Getting the type of 'den' (line 76)
        den_292566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 33), 'den', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292564, den_292566)
        # Adding element type (line 76)
        float_292567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292564, float_292567)
        
        
        # Obtaining the type of the subscript
        slice_292568 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 44), None, None, None)
        int_292569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 49), 'int')
        # Getting the type of 'u' (line 76)
        u_292570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___292571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 44), u_292570, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_292572 = invoke(stypy.reporting.localization.Localization(__file__, 76, 44), getitem___292571, (slice_292568, int_292569))
        
        # Getting the type of 't_in' (line 76)
        t_in_292573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 53), 't_in', False)
        # Processing the call keyword arguments (line 76)
        kwargs_292574 = {}
        # Getting the type of 'dlsim' (line 76)
        dlsim_292563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 76)
        dlsim_call_result_292575 = invoke(stypy.reporting.localization.Localization(__file__, 76, 21), dlsim_292563, *[tuple_292564, subscript_call_result_292572, t_in_292573], **kwargs_292574)
        
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___292576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), dlsim_call_result_292575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_292577 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___292576, int_292562)
        
        # Assigning a type to the variable 'tuple_var_assignment_292177' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_292177', subscript_call_result_292577)
        
        # Assigning a Subscript to a Name (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_292578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'int')
        
        # Call to dlsim(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_292580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'num' (line 76)
        num_292581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 28), 'num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292580, num_292581)
        # Adding element type (line 76)
        # Getting the type of 'den' (line 76)
        den_292582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 33), 'den', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292580, den_292582)
        # Adding element type (line 76)
        float_292583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 28), tuple_292580, float_292583)
        
        
        # Obtaining the type of the subscript
        slice_292584 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 76, 44), None, None, None)
        int_292585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 49), 'int')
        # Getting the type of 'u' (line 76)
        u_292586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 44), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___292587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 44), u_292586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_292588 = invoke(stypy.reporting.localization.Localization(__file__, 76, 44), getitem___292587, (slice_292584, int_292585))
        
        # Getting the type of 't_in' (line 76)
        t_in_292589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 53), 't_in', False)
        # Processing the call keyword arguments (line 76)
        kwargs_292590 = {}
        # Getting the type of 'dlsim' (line 76)
        dlsim_292579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 76)
        dlsim_call_result_292591 = invoke(stypy.reporting.localization.Localization(__file__, 76, 21), dlsim_292579, *[tuple_292580, subscript_call_result_292588, t_in_292589], **kwargs_292590)
        
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___292592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), dlsim_call_result_292591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_292593 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), getitem___292592, int_292578)
        
        # Assigning a type to the variable 'tuple_var_assignment_292178' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_292178', subscript_call_result_292593)
        
        # Assigning a Name to a Name (line 76):
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_292177' (line 76)
        tuple_var_assignment_292177_292594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_292177')
        # Assigning a type to the variable 'tout' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tout', tuple_var_assignment_292177_292594)
        
        # Assigning a Name to a Name (line 76):
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_292178' (line 76)
        tuple_var_assignment_292178_292595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'tuple_var_assignment_292178')
        # Assigning a type to the variable 'yout' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'yout', tuple_var_assignment_292178_292595)
        
        # Call to assert_array_almost_equal(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'yout' (line 78)
        yout_292597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'yout', False)
        # Getting the type of 'yout_truth' (line 78)
        yout_truth_292598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'yout_truth', False)
        # Processing the call keyword arguments (line 78)
        kwargs_292599 = {}
        # Getting the type of 'assert_array_almost_equal' (line 78)
        assert_array_almost_equal_292596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 78)
        assert_array_almost_equal_call_result_292600 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), assert_array_almost_equal_292596, *[yout_292597, yout_truth_292598], **kwargs_292599)
        
        
        # Call to assert_array_almost_equal(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 't_in' (line 79)
        t_in_292602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 't_in', False)
        # Getting the type of 'tout' (line 79)
        tout_292603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'tout', False)
        # Processing the call keyword arguments (line 79)
        kwargs_292604 = {}
        # Getting the type of 'assert_array_almost_equal' (line 79)
        assert_array_almost_equal_292601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 79)
        assert_array_almost_equal_call_result_292605 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_array_almost_equal_292601, *[t_in_292602, tout_292603], **kwargs_292604)
        
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to asarray(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Obtaining the type of the subscript
        slice_292608 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 27), None, None, None)
        int_292609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 32), 'int')
        # Getting the type of 'u' (line 82)
        u_292610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___292611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), u_292610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_292612 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), getitem___292611, (slice_292608, int_292609))
        
        # Processing the call keyword arguments (line 82)
        kwargs_292613 = {}
        # Getting the type of 'np' (line 82)
        np_292606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'np', False)
        # Obtaining the member 'asarray' of a type (line 82)
        asarray_292607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 16), np_292606, 'asarray')
        # Calling asarray(args, kwargs) (line 82)
        asarray_call_result_292614 = invoke(stypy.reporting.localization.Localization(__file__, 82, 16), asarray_292607, *[subscript_call_result_292612], **kwargs_292613)
        
        # Assigning a type to the variable 'uflat' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'uflat', asarray_call_result_292614)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to reshape(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_292617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        int_292618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 31), tuple_292617, int_292618)
        
        # Processing the call keyword arguments (line 83)
        kwargs_292619 = {}
        # Getting the type of 'uflat' (line 83)
        uflat_292615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'uflat', False)
        # Obtaining the member 'reshape' of a type (line 83)
        reshape_292616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), uflat_292615, 'reshape')
        # Calling reshape(args, kwargs) (line 83)
        reshape_call_result_292620 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), reshape_292616, *[tuple_292617], **kwargs_292619)
        
        # Assigning a type to the variable 'uflat' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'uflat', reshape_call_result_292620)
        
        # Assigning a Call to a Tuple (line 84):
        
        # Assigning a Subscript to a Name (line 84):
        
        # Assigning a Subscript to a Name (line 84):
        
        # Obtaining the type of the subscript
        int_292621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        
        # Call to dlsim(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_292623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        # Getting the type of 'num' (line 84)
        num_292624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292623, num_292624)
        # Adding element type (line 84)
        # Getting the type of 'den' (line 84)
        den_292625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'den', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292623, den_292625)
        # Adding element type (line 84)
        float_292626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292623, float_292626)
        
        # Getting the type of 'uflat' (line 84)
        uflat_292627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 44), 'uflat', False)
        # Getting the type of 't_in' (line 84)
        t_in_292628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 't_in', False)
        # Processing the call keyword arguments (line 84)
        kwargs_292629 = {}
        # Getting the type of 'dlsim' (line 84)
        dlsim_292622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 84)
        dlsim_call_result_292630 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), dlsim_292622, *[tuple_292623, uflat_292627, t_in_292628], **kwargs_292629)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___292631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), dlsim_call_result_292630, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_292632 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___292631, int_292621)
        
        # Assigning a type to the variable 'tuple_var_assignment_292179' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_292179', subscript_call_result_292632)
        
        # Assigning a Subscript to a Name (line 84):
        
        # Assigning a Subscript to a Name (line 84):
        
        # Obtaining the type of the subscript
        int_292633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'int')
        
        # Call to dlsim(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_292635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        # Getting the type of 'num' (line 84)
        num_292636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292635, num_292636)
        # Adding element type (line 84)
        # Getting the type of 'den' (line 84)
        den_292637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'den', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292635, den_292637)
        # Adding element type (line 84)
        float_292638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 28), tuple_292635, float_292638)
        
        # Getting the type of 'uflat' (line 84)
        uflat_292639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 44), 'uflat', False)
        # Getting the type of 't_in' (line 84)
        t_in_292640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 't_in', False)
        # Processing the call keyword arguments (line 84)
        kwargs_292641 = {}
        # Getting the type of 'dlsim' (line 84)
        dlsim_292634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 84)
        dlsim_call_result_292642 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), dlsim_292634, *[tuple_292635, uflat_292639, t_in_292640], **kwargs_292641)
        
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___292643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), dlsim_call_result_292642, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_292644 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), getitem___292643, int_292633)
        
        # Assigning a type to the variable 'tuple_var_assignment_292180' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_292180', subscript_call_result_292644)
        
        # Assigning a Name to a Name (line 84):
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'tuple_var_assignment_292179' (line 84)
        tuple_var_assignment_292179_292645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_292179')
        # Assigning a type to the variable 'tout' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tout', tuple_var_assignment_292179_292645)
        
        # Assigning a Name to a Name (line 84):
        
        # Assigning a Name to a Name (line 84):
        # Getting the type of 'tuple_var_assignment_292180' (line 84)
        tuple_var_assignment_292180_292646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'tuple_var_assignment_292180')
        # Assigning a type to the variable 'yout' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'yout', tuple_var_assignment_292180_292646)
        
        # Call to assert_array_almost_equal(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'yout' (line 86)
        yout_292648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'yout', False)
        # Getting the type of 'yout_truth' (line 86)
        yout_truth_292649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 40), 'yout_truth', False)
        # Processing the call keyword arguments (line 86)
        kwargs_292650 = {}
        # Getting the type of 'assert_array_almost_equal' (line 86)
        assert_array_almost_equal_292647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 86)
        assert_array_almost_equal_call_result_292651 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), assert_array_almost_equal_292647, *[yout_292648, yout_truth_292649], **kwargs_292650)
        
        
        # Call to assert_array_almost_equal(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 't_in' (line 87)
        t_in_292653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 't_in', False)
        # Getting the type of 'tout' (line 87)
        tout_292654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'tout', False)
        # Processing the call keyword arguments (line 87)
        kwargs_292655 = {}
        # Getting the type of 'assert_array_almost_equal' (line 87)
        assert_array_almost_equal_292652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 87)
        assert_array_almost_equal_call_result_292656 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), assert_array_almost_equal_292652, *[t_in_292653, tout_292654], **kwargs_292655)
        
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_292659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_292660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 22), list_292659, float_292660)
        # Adding element type (line 90)
        float_292661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 22), list_292659, float_292661)
        
        # Processing the call keyword arguments (line 90)
        kwargs_292662 = {}
        # Getting the type of 'np' (line 90)
        np_292657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_292658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), np_292657, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_292663 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), array_292658, *[list_292659], **kwargs_292662)
        
        # Assigning a type to the variable 'zd' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'zd', array_call_result_292663)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to array(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_292666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        complex_292667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 23), 'complex')
        
        # Call to sqrt(...): (line 91)
        # Processing the call arguments (line 91)
        int_292670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_292671 = {}
        # Getting the type of 'np' (line 91)
        np_292668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 91)
        sqrt_292669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), np_292668, 'sqrt')
        # Calling sqrt(args, kwargs) (line 91)
        sqrt_call_result_292672 = invoke(stypy.reporting.localization.Localization(__file__, 91, 29), sqrt_292669, *[int_292670], **kwargs_292671)
        
        # Applying the binary operator 'div' (line 91)
        result_div_292673 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 23), 'div', complex_292667, sqrt_call_result_292672)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 22), list_292666, result_div_292673)
        # Adding element type (line 91)
        complex_292674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 41), 'complex')
        
        # Call to sqrt(...): (line 91)
        # Processing the call arguments (line 91)
        int_292677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 56), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_292678 = {}
        # Getting the type of 'np' (line 91)
        np_292675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 91)
        sqrt_292676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 48), np_292675, 'sqrt')
        # Calling sqrt(args, kwargs) (line 91)
        sqrt_call_result_292679 = invoke(stypy.reporting.localization.Localization(__file__, 91, 48), sqrt_292676, *[int_292677], **kwargs_292678)
        
        # Applying the binary operator 'div' (line 91)
        result_div_292680 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 41), 'div', complex_292674, sqrt_call_result_292679)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 22), list_292666, result_div_292680)
        
        # Processing the call keyword arguments (line 91)
        kwargs_292681 = {}
        # Getting the type of 'np' (line 91)
        np_292664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 91)
        array_292665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), np_292664, 'array')
        # Calling array(args, kwargs) (line 91)
        array_call_result_292682 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), array_292665, *[list_292666], **kwargs_292681)
        
        # Assigning a type to the variable 'pd' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'pd', array_call_result_292682)
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        float_292683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'float')
        # Assigning a type to the variable 'k' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'k', float_292683)
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to transpose(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_292695 = {}
        
        # Call to asmatrix(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_292686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        float_292687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 33), list_292686, float_292687)
        # Adding element type (line 93)
        float_292688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 33), list_292686, float_292688)
        # Adding element type (line 93)
        float_292689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 33), list_292686, float_292689)
        # Adding element type (line 93)
        float_292690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 33), list_292686, float_292690)
        # Adding element type (line 93)
        float_292691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 33), list_292686, float_292691)
        
        # Processing the call keyword arguments (line 93)
        kwargs_292692 = {}
        # Getting the type of 'np' (line 93)
        np_292684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 93)
        asmatrix_292685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), np_292684, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 93)
        asmatrix_call_result_292693 = invoke(stypy.reporting.localization.Localization(__file__, 93, 21), asmatrix_292685, *[list_292686], **kwargs_292692)
        
        # Obtaining the member 'transpose' of a type (line 93)
        transpose_292694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), asmatrix_call_result_292693, 'transpose')
        # Calling transpose(args, kwargs) (line 93)
        transpose_call_result_292696 = invoke(stypy.reporting.localization.Localization(__file__, 93, 21), transpose_292694, *[], **kwargs_292695)
        
        # Assigning a type to the variable 'yout_truth' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'yout_truth', transpose_call_result_292696)
        
        # Assigning a Call to a Tuple (line 95):
        
        # Assigning a Subscript to a Name (line 95):
        
        # Assigning a Subscript to a Name (line 95):
        
        # Obtaining the type of the subscript
        int_292697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
        
        # Call to dlsim(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_292699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'zd' (line 95)
        zd_292700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'zd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292699, zd_292700)
        # Adding element type (line 95)
        # Getting the type of 'pd' (line 95)
        pd_292701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'pd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292699, pd_292701)
        # Adding element type (line 95)
        # Getting the type of 'k' (line 95)
        k_292702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292699, k_292702)
        # Adding element type (line 95)
        float_292703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292699, float_292703)
        
        
        # Obtaining the type of the subscript
        slice_292704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 45), None, None, None)
        int_292705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 50), 'int')
        # Getting the type of 'u' (line 95)
        u_292706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 45), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___292707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 45), u_292706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_292708 = invoke(stypy.reporting.localization.Localization(__file__, 95, 45), getitem___292707, (slice_292704, int_292705))
        
        # Getting the type of 't_in' (line 95)
        t_in_292709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 't_in', False)
        # Processing the call keyword arguments (line 95)
        kwargs_292710 = {}
        # Getting the type of 'dlsim' (line 95)
        dlsim_292698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 95)
        dlsim_call_result_292711 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), dlsim_292698, *[tuple_292699, subscript_call_result_292708, t_in_292709], **kwargs_292710)
        
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___292712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), dlsim_call_result_292711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_292713 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), getitem___292712, int_292697)
        
        # Assigning a type to the variable 'tuple_var_assignment_292181' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_292181', subscript_call_result_292713)
        
        # Assigning a Subscript to a Name (line 95):
        
        # Assigning a Subscript to a Name (line 95):
        
        # Obtaining the type of the subscript
        int_292714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
        
        # Call to dlsim(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_292716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        # Getting the type of 'zd' (line 95)
        zd_292717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'zd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292716, zd_292717)
        # Adding element type (line 95)
        # Getting the type of 'pd' (line 95)
        pd_292718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'pd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292716, pd_292718)
        # Adding element type (line 95)
        # Getting the type of 'k' (line 95)
        k_292719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292716, k_292719)
        # Adding element type (line 95)
        float_292720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 28), tuple_292716, float_292720)
        
        
        # Obtaining the type of the subscript
        slice_292721 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 95, 45), None, None, None)
        int_292722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 50), 'int')
        # Getting the type of 'u' (line 95)
        u_292723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 45), 'u', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___292724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 45), u_292723, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_292725 = invoke(stypy.reporting.localization.Localization(__file__, 95, 45), getitem___292724, (slice_292721, int_292722))
        
        # Getting the type of 't_in' (line 95)
        t_in_292726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 't_in', False)
        # Processing the call keyword arguments (line 95)
        kwargs_292727 = {}
        # Getting the type of 'dlsim' (line 95)
        dlsim_292715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 95)
        dlsim_call_result_292728 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), dlsim_292715, *[tuple_292716, subscript_call_result_292725, t_in_292726], **kwargs_292727)
        
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___292729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), dlsim_call_result_292728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 95)
        subscript_call_result_292730 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), getitem___292729, int_292714)
        
        # Assigning a type to the variable 'tuple_var_assignment_292182' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_292182', subscript_call_result_292730)
        
        # Assigning a Name to a Name (line 95):
        
        # Assigning a Name to a Name (line 95):
        # Getting the type of 'tuple_var_assignment_292181' (line 95)
        tuple_var_assignment_292181_292731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_292181')
        # Assigning a type to the variable 'tout' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tout', tuple_var_assignment_292181_292731)
        
        # Assigning a Name to a Name (line 95):
        
        # Assigning a Name to a Name (line 95):
        # Getting the type of 'tuple_var_assignment_292182' (line 95)
        tuple_var_assignment_292182_292732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'tuple_var_assignment_292182')
        # Assigning a type to the variable 'yout' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 14), 'yout', tuple_var_assignment_292182_292732)
        
        # Call to assert_array_almost_equal(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'yout' (line 97)
        yout_292734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 34), 'yout', False)
        # Getting the type of 'yout_truth' (line 97)
        yout_truth_292735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'yout_truth', False)
        # Processing the call keyword arguments (line 97)
        kwargs_292736 = {}
        # Getting the type of 'assert_array_almost_equal' (line 97)
        assert_array_almost_equal_292733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 97)
        assert_array_almost_equal_call_result_292737 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_array_almost_equal_292733, *[yout_292734, yout_truth_292735], **kwargs_292736)
        
        
        # Call to assert_array_almost_equal(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 't_in' (line 98)
        t_in_292739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 't_in', False)
        # Getting the type of 'tout' (line 98)
        tout_292740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 40), 'tout', False)
        # Processing the call keyword arguments (line 98)
        kwargs_292741 = {}
        # Getting the type of 'assert_array_almost_equal' (line 98)
        assert_array_almost_equal_292738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 98)
        assert_array_almost_equal_call_result_292742 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), assert_array_almost_equal_292738, *[t_in_292739, tout_292740], **kwargs_292741)
        
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to lti(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_292744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_292745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 21), list_292744, int_292745)
        
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_292746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        int_292747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), list_292746, int_292747)
        # Adding element type (line 101)
        int_292748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 26), list_292746, int_292748)
        
        # Processing the call keyword arguments (line 101)
        kwargs_292749 = {}
        # Getting the type of 'lti' (line 101)
        lti_292743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'lti', False)
        # Calling lti(args, kwargs) (line 101)
        lti_call_result_292750 = invoke(stypy.reporting.localization.Localization(__file__, 101, 17), lti_292743, *[list_292744, list_292746], **kwargs_292749)
        
        # Assigning a type to the variable 'system' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'system', lti_call_result_292750)
        
        # Call to assert_raises(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'AttributeError' (line 102)
        AttributeError_292752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'AttributeError', False)
        # Getting the type of 'dlsim' (line 102)
        dlsim_292753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'dlsim', False)
        # Getting the type of 'system' (line 102)
        system_292754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'system', False)
        # Getting the type of 'u' (line 102)
        u_292755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'u', False)
        # Processing the call keyword arguments (line 102)
        kwargs_292756 = {}
        # Getting the type of 'assert_raises' (line 102)
        assert_raises_292751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 102)
        assert_raises_call_result_292757 = invoke(stypy.reporting.localization.Localization(__file__, 102, 8), assert_raises_292751, *[AttributeError_292752, dlsim_292753, system_292754, u_292755], **kwargs_292756)
        
        
        # ################# End of 'test_dlsim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlsim' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_292758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292758)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlsim'
        return stypy_return_type_292758


    @norecursion
    def test_dstep(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dstep'
        module_type_store = module_type_store.open_function_context('test_dstep', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dstep')
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dstep.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dstep', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dstep', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dstep(...)' code ##################

        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to asarray(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_292761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_292762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        float_292763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 24), list_292762, float_292763)
        # Adding element type (line 106)
        float_292764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 24), list_292762, float_292764)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), list_292761, list_292762)
        # Adding element type (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_292765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        float_292766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 36), list_292765, float_292766)
        # Adding element type (line 106)
        float_292767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 36), list_292765, float_292767)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 23), list_292761, list_292765)
        
        # Processing the call keyword arguments (line 106)
        kwargs_292768 = {}
        # Getting the type of 'np' (line 106)
        np_292759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 106)
        asarray_292760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), np_292759, 'asarray')
        # Calling asarray(args, kwargs) (line 106)
        asarray_call_result_292769 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), asarray_292760, *[list_292761], **kwargs_292768)
        
        # Assigning a type to the variable 'a' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'a', asarray_call_result_292769)
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to asarray(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_292772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_292773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        float_292774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 24), list_292773, float_292774)
        # Adding element type (line 107)
        float_292775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 24), list_292773, float_292775)
        # Adding element type (line 107)
        float_292776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 24), list_292773, float_292776)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), list_292772, list_292773)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_292777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        float_292778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 42), list_292777, float_292778)
        # Adding element type (line 107)
        float_292779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 42), list_292777, float_292779)
        # Adding element type (line 107)
        float_292780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 42), list_292777, float_292780)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), list_292772, list_292777)
        
        # Processing the call keyword arguments (line 107)
        kwargs_292781 = {}
        # Getting the type of 'np' (line 107)
        np_292770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 107)
        asarray_292771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), np_292770, 'asarray')
        # Calling asarray(args, kwargs) (line 107)
        asarray_call_result_292782 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), asarray_292771, *[list_292772], **kwargs_292781)
        
        # Assigning a type to the variable 'b' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'b', asarray_call_result_292782)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to asarray(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_292785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_292786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        float_292787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 24), list_292786, float_292787)
        # Adding element type (line 108)
        float_292788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 24), list_292786, float_292788)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 23), list_292785, list_292786)
        
        # Processing the call keyword arguments (line 108)
        kwargs_292789 = {}
        # Getting the type of 'np' (line 108)
        np_292783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 108)
        asarray_292784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), np_292783, 'asarray')
        # Calling asarray(args, kwargs) (line 108)
        asarray_call_result_292790 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), asarray_292784, *[list_292785], **kwargs_292789)
        
        # Assigning a type to the variable 'c' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'c', asarray_call_result_292790)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to asarray(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_292793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_292794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_292795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_292794, float_292795)
        # Adding element type (line 109)
        float_292796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_292794, float_292796)
        # Adding element type (line 109)
        float_292797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 24), list_292794, float_292797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 23), list_292793, list_292794)
        
        # Processing the call keyword arguments (line 109)
        kwargs_292798 = {}
        # Getting the type of 'np' (line 109)
        np_292791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 109)
        asarray_292792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), np_292791, 'asarray')
        # Calling asarray(args, kwargs) (line 109)
        asarray_call_result_292799 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), asarray_292792, *[list_292793], **kwargs_292798)
        
        # Assigning a type to the variable 'd' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'd', asarray_call_result_292799)
        
        # Assigning a Num to a Name (line 110):
        
        # Assigning a Num to a Name (line 110):
        
        # Assigning a Num to a Name (line 110):
        float_292800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 13), 'float')
        # Assigning a type to the variable 'dt' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'dt', float_292800)
        
        # Assigning a Tuple to a Name (line 114):
        
        # Assigning a Tuple to a Name (line 114):
        
        # Assigning a Tuple to a Name (line 114):
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_292801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        
        # Call to asarray(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_292804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_292805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292805)
        # Adding element type (line 114)
        float_292806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292806)
        # Adding element type (line 114)
        float_292807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292807)
        # Adding element type (line 114)
        float_292808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292808)
        # Adding element type (line 114)
        float_292809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292809)
        # Adding element type (line 114)
        float_292810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292810)
        # Adding element type (line 114)
        float_292811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292811)
        # Adding element type (line 114)
        float_292812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292812)
        # Adding element type (line 114)
        float_292813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292813)
        # Adding element type (line 114)
        float_292814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 38), list_292804, float_292814)
        
        # Processing the call keyword arguments (line 114)
        kwargs_292815 = {}
        # Getting the type of 'np' (line 114)
        np_292802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'np', False)
        # Obtaining the member 'asarray' of a type (line 114)
        asarray_292803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 27), np_292802, 'asarray')
        # Calling asarray(args, kwargs) (line 114)
        asarray_call_result_292816 = invoke(stypy.reporting.localization.Localization(__file__, 114, 27), asarray_292803, *[list_292804], **kwargs_292815)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), tuple_292801, asarray_call_result_292816)
        # Adding element type (line 114)
        
        # Call to asarray(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_292819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        float_292820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292820)
        # Adding element type (line 117)
        float_292821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292821)
        # Adding element type (line 117)
        float_292822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292822)
        # Adding element type (line 117)
        float_292823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292823)
        # Adding element type (line 117)
        float_292824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292824)
        # Adding element type (line 117)
        float_292825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292825)
        # Adding element type (line 117)
        float_292826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292826)
        # Adding element type (line 117)
        float_292827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292827)
        # Adding element type (line 117)
        float_292828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292828)
        # Adding element type (line 117)
        float_292829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 38), list_292819, float_292829)
        
        # Processing the call keyword arguments (line 117)
        kwargs_292830 = {}
        # Getting the type of 'np' (line 117)
        np_292817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'np', False)
        # Obtaining the member 'asarray' of a type (line 117)
        asarray_292818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 27), np_292817, 'asarray')
        # Calling asarray(args, kwargs) (line 117)
        asarray_call_result_292831 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), asarray_292818, *[list_292819], **kwargs_292830)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), tuple_292801, asarray_call_result_292831)
        # Adding element type (line 114)
        
        # Call to asarray(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_292834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        float_292835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292835)
        # Adding element type (line 121)
        float_292836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292836)
        # Adding element type (line 121)
        float_292837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292837)
        # Adding element type (line 121)
        float_292838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292838)
        # Adding element type (line 121)
        float_292839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292839)
        # Adding element type (line 121)
        float_292840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292840)
        # Adding element type (line 121)
        float_292841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292841)
        # Adding element type (line 121)
        float_292842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292842)
        # Adding element type (line 121)
        float_292843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292843)
        # Adding element type (line 121)
        float_292844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 38), list_292834, float_292844)
        
        # Processing the call keyword arguments (line 121)
        kwargs_292845 = {}
        # Getting the type of 'np' (line 121)
        np_292832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'np', False)
        # Obtaining the member 'asarray' of a type (line 121)
        asarray_292833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 27), np_292832, 'asarray')
        # Calling asarray(args, kwargs) (line 121)
        asarray_call_result_292846 = invoke(stypy.reporting.localization.Localization(__file__, 121, 27), asarray_292833, *[list_292834], **kwargs_292845)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), tuple_292801, asarray_call_result_292846)
        
        # Assigning a type to the variable 'yout_step_truth' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'yout_step_truth', tuple_292801)
        
        # Assigning a Call to a Tuple (line 125):
        
        # Assigning a Subscript to a Name (line 125):
        
        # Assigning a Subscript to a Name (line 125):
        
        # Obtaining the type of the subscript
        int_292847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'int')
        
        # Call to dstep(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_292849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'a' (line 125)
        a_292850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292849, a_292850)
        # Adding element type (line 125)
        # Getting the type of 'b' (line 125)
        b_292851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292849, b_292851)
        # Adding element type (line 125)
        # Getting the type of 'c' (line 125)
        c_292852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292849, c_292852)
        # Adding element type (line 125)
        # Getting the type of 'd' (line 125)
        d_292853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292849, d_292853)
        # Adding element type (line 125)
        # Getting the type of 'dt' (line 125)
        dt_292854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292849, dt_292854)
        
        # Processing the call keyword arguments (line 125)
        int_292855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'int')
        keyword_292856 = int_292855
        kwargs_292857 = {'n': keyword_292856}
        # Getting the type of 'dstep' (line 125)
        dstep_292848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 125)
        dstep_call_result_292858 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), dstep_292848, *[tuple_292849], **kwargs_292857)
        
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___292859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), dstep_call_result_292858, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_292860 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), getitem___292859, int_292847)
        
        # Assigning a type to the variable 'tuple_var_assignment_292183' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'tuple_var_assignment_292183', subscript_call_result_292860)
        
        # Assigning a Subscript to a Name (line 125):
        
        # Assigning a Subscript to a Name (line 125):
        
        # Obtaining the type of the subscript
        int_292861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 8), 'int')
        
        # Call to dstep(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_292863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        # Getting the type of 'a' (line 125)
        a_292864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292863, a_292864)
        # Adding element type (line 125)
        # Getting the type of 'b' (line 125)
        b_292865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292863, b_292865)
        # Adding element type (line 125)
        # Getting the type of 'c' (line 125)
        c_292866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292863, c_292866)
        # Adding element type (line 125)
        # Getting the type of 'd' (line 125)
        d_292867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292863, d_292867)
        # Adding element type (line 125)
        # Getting the type of 'dt' (line 125)
        dt_292868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), tuple_292863, dt_292868)
        
        # Processing the call keyword arguments (line 125)
        int_292869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 47), 'int')
        keyword_292870 = int_292869
        kwargs_292871 = {'n': keyword_292870}
        # Getting the type of 'dstep' (line 125)
        dstep_292862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 125)
        dstep_call_result_292872 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), dstep_292862, *[tuple_292863], **kwargs_292871)
        
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___292873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), dstep_call_result_292872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_292874 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), getitem___292873, int_292861)
        
        # Assigning a type to the variable 'tuple_var_assignment_292184' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'tuple_var_assignment_292184', subscript_call_result_292874)
        
        # Assigning a Name to a Name (line 125):
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'tuple_var_assignment_292183' (line 125)
        tuple_var_assignment_292183_292875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'tuple_var_assignment_292183')
        # Assigning a type to the variable 'tout' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'tout', tuple_var_assignment_292183_292875)
        
        # Assigning a Name to a Name (line 125):
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'tuple_var_assignment_292184' (line 125)
        tuple_var_assignment_292184_292876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'tuple_var_assignment_292184')
        # Assigning a type to the variable 'yout' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 14), 'yout', tuple_var_assignment_292184_292876)
        
        # Call to assert_equal(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to len(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'yout' (line 127)
        yout_292879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 'yout', False)
        # Processing the call keyword arguments (line 127)
        kwargs_292880 = {}
        # Getting the type of 'len' (line 127)
        len_292878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'len', False)
        # Calling len(args, kwargs) (line 127)
        len_call_result_292881 = invoke(stypy.reporting.localization.Localization(__file__, 127, 21), len_292878, *[yout_292879], **kwargs_292880)
        
        int_292882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 32), 'int')
        # Processing the call keyword arguments (line 127)
        kwargs_292883 = {}
        # Getting the type of 'assert_equal' (line 127)
        assert_equal_292877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 127)
        assert_equal_call_result_292884 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assert_equal_292877, *[len_call_result_292881, int_292882], **kwargs_292883)
        
        
        
        # Call to range(...): (line 129)
        # Processing the call arguments (line 129)
        int_292886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'int')
        
        # Call to len(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'yout' (line 129)
        yout_292888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'yout', False)
        # Processing the call keyword arguments (line 129)
        kwargs_292889 = {}
        # Getting the type of 'len' (line 129)
        len_292887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'len', False)
        # Calling len(args, kwargs) (line 129)
        len_call_result_292890 = invoke(stypy.reporting.localization.Localization(__file__, 129, 26), len_292887, *[yout_292888], **kwargs_292889)
        
        # Processing the call keyword arguments (line 129)
        kwargs_292891 = {}
        # Getting the type of 'range' (line 129)
        range_292885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'range', False)
        # Calling range(args, kwargs) (line 129)
        range_call_result_292892 = invoke(stypy.reporting.localization.Localization(__file__, 129, 17), range_292885, *[int_292886, len_call_result_292890], **kwargs_292891)
        
        # Testing the type of a for loop iterable (line 129)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 129, 8), range_call_result_292892)
        # Getting the type of the for loop variable (line 129)
        for_loop_var_292893 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 129, 8), range_call_result_292892)
        # Assigning a type to the variable 'i' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'i', for_loop_var_292893)
        # SSA begins for a for statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Obtaining the type of the subscript
        int_292895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 130)
        i_292896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'i', False)
        # Getting the type of 'yout' (line 130)
        yout_292897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___292898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), yout_292897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_292899 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), getitem___292898, i_292896)
        
        # Obtaining the member 'shape' of a type (line 130)
        shape_292900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), subscript_call_result_292899, 'shape')
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___292901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), shape_292900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_292902 = invoke(stypy.reporting.localization.Localization(__file__, 130, 25), getitem___292901, int_292895)
        
        int_292903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 43), 'int')
        # Processing the call keyword arguments (line 130)
        kwargs_292904 = {}
        # Getting the type of 'assert_equal' (line 130)
        assert_equal_292894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 130)
        assert_equal_call_result_292905 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), assert_equal_292894, *[subscript_call_result_292902, int_292903], **kwargs_292904)
        
        
        # Call to assert_array_almost_equal(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to flatten(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_292912 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 131)
        i_292907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 43), 'i', False)
        # Getting the type of 'yout' (line 131)
        yout_292908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 38), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___292909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 38), yout_292908, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_292910 = invoke(stypy.reporting.localization.Localization(__file__, 131, 38), getitem___292909, i_292907)
        
        # Obtaining the member 'flatten' of a type (line 131)
        flatten_292911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 38), subscript_call_result_292910, 'flatten')
        # Calling flatten(args, kwargs) (line 131)
        flatten_call_result_292913 = invoke(stypy.reporting.localization.Localization(__file__, 131, 38), flatten_292911, *[], **kwargs_292912)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 131)
        i_292914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 73), 'i', False)
        # Getting the type of 'yout_step_truth' (line 131)
        yout_step_truth_292915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 57), 'yout_step_truth', False)
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___292916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 57), yout_step_truth_292915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_292917 = invoke(stypy.reporting.localization.Localization(__file__, 131, 57), getitem___292916, i_292914)
        
        # Processing the call keyword arguments (line 131)
        kwargs_292918 = {}
        # Getting the type of 'assert_array_almost_equal' (line 131)
        assert_array_almost_equal_292906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 131)
        assert_array_almost_equal_call_result_292919 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), assert_array_almost_equal_292906, *[flatten_call_result_292913, subscript_call_result_292917], **kwargs_292918)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 134):
        
        # Assigning a Tuple to a Name (line 134):
        
        # Assigning a Tuple to a Name (line 134):
        
        # Obtaining an instance of the builtin type 'tuple' (line 134)
        tuple_292920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 134)
        # Adding element type (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_292921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        float_292922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 16), list_292921, float_292922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 16), tuple_292920, list_292921)
        # Adding element type (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_292923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        float_292924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 23), list_292923, float_292924)
        # Adding element type (line 134)
        float_292925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 23), list_292923, float_292925)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 16), tuple_292920, list_292923)
        # Adding element type (line 134)
        float_292926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 16), tuple_292920, float_292926)
        
        # Assigning a type to the variable 'tfin' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tfin', tuple_292920)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to asarray(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_292929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        float_292930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 33), list_292929, float_292930)
        # Adding element type (line 135)
        float_292931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 33), list_292929, float_292931)
        # Adding element type (line 135)
        float_292932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 33), list_292929, float_292932)
        
        # Processing the call keyword arguments (line 135)
        kwargs_292933 = {}
        # Getting the type of 'np' (line 135)
        np_292927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'np', False)
        # Obtaining the member 'asarray' of a type (line 135)
        asarray_292928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 22), np_292927, 'asarray')
        # Calling asarray(args, kwargs) (line 135)
        asarray_call_result_292934 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), asarray_292928, *[list_292929], **kwargs_292933)
        
        # Assigning a type to the variable 'yout_tfstep' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'yout_tfstep', asarray_call_result_292934)
        
        # Assigning a Call to a Tuple (line 136):
        
        # Assigning a Subscript to a Name (line 136):
        
        # Assigning a Subscript to a Name (line 136):
        
        # Obtaining the type of the subscript
        int_292935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
        
        # Call to dstep(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'tfin' (line 136)
        tfin_292937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'tfin', False)
        # Processing the call keyword arguments (line 136)
        int_292938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'int')
        keyword_292939 = int_292938
        kwargs_292940 = {'n': keyword_292939}
        # Getting the type of 'dstep' (line 136)
        dstep_292936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 136)
        dstep_call_result_292941 = invoke(stypy.reporting.localization.Localization(__file__, 136, 21), dstep_292936, *[tfin_292937], **kwargs_292940)
        
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___292942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), dstep_call_result_292941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_292943 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___292942, int_292935)
        
        # Assigning a type to the variable 'tuple_var_assignment_292185' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_292185', subscript_call_result_292943)
        
        # Assigning a Subscript to a Name (line 136):
        
        # Assigning a Subscript to a Name (line 136):
        
        # Obtaining the type of the subscript
        int_292944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
        
        # Call to dstep(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'tfin' (line 136)
        tfin_292946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'tfin', False)
        # Processing the call keyword arguments (line 136)
        int_292947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'int')
        keyword_292948 = int_292947
        kwargs_292949 = {'n': keyword_292948}
        # Getting the type of 'dstep' (line 136)
        dstep_292945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 136)
        dstep_call_result_292950 = invoke(stypy.reporting.localization.Localization(__file__, 136, 21), dstep_292945, *[tfin_292946], **kwargs_292949)
        
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___292951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), dstep_call_result_292950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_292952 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___292951, int_292944)
        
        # Assigning a type to the variable 'tuple_var_assignment_292186' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_292186', subscript_call_result_292952)
        
        # Assigning a Name to a Name (line 136):
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_var_assignment_292185' (line 136)
        tuple_var_assignment_292185_292953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_292185')
        # Assigning a type to the variable 'tout' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tout', tuple_var_assignment_292185_292953)
        
        # Assigning a Name to a Name (line 136):
        
        # Assigning a Name to a Name (line 136):
        # Getting the type of 'tuple_var_assignment_292186' (line 136)
        tuple_var_assignment_292186_292954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_292186')
        # Assigning a type to the variable 'yout' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'yout', tuple_var_assignment_292186_292954)
        
        # Call to assert_equal(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'yout' (line 137)
        yout_292957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'yout', False)
        # Processing the call keyword arguments (line 137)
        kwargs_292958 = {}
        # Getting the type of 'len' (line 137)
        len_292956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_292959 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), len_292956, *[yout_292957], **kwargs_292958)
        
        int_292960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 32), 'int')
        # Processing the call keyword arguments (line 137)
        kwargs_292961 = {}
        # Getting the type of 'assert_equal' (line 137)
        assert_equal_292955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 137)
        assert_equal_call_result_292962 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_equal_292955, *[len_call_result_292959, int_292960], **kwargs_292961)
        
        
        # Call to assert_array_almost_equal(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to flatten(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_292969 = {}
        
        # Obtaining the type of the subscript
        int_292964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 39), 'int')
        # Getting the type of 'yout' (line 138)
        yout_292965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___292966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 34), yout_292965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_292967 = invoke(stypy.reporting.localization.Localization(__file__, 138, 34), getitem___292966, int_292964)
        
        # Obtaining the member 'flatten' of a type (line 138)
        flatten_292968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 34), subscript_call_result_292967, 'flatten')
        # Calling flatten(args, kwargs) (line 138)
        flatten_call_result_292970 = invoke(stypy.reporting.localization.Localization(__file__, 138, 34), flatten_292968, *[], **kwargs_292969)
        
        # Getting the type of 'yout_tfstep' (line 138)
        yout_tfstep_292971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 53), 'yout_tfstep', False)
        # Processing the call keyword arguments (line 138)
        kwargs_292972 = {}
        # Getting the type of 'assert_array_almost_equal' (line 138)
        assert_array_almost_equal_292963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 138)
        assert_array_almost_equal_call_result_292973 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assert_array_almost_equal_292963, *[flatten_call_result_292970, yout_tfstep_292971], **kwargs_292972)
        
        
        # Assigning a BinOp to a Name (line 140):
        
        # Assigning a BinOp to a Name (line 140):
        
        # Assigning a BinOp to a Name (line 140):
        
        # Call to tf2zpk(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining the type of the subscript
        int_292975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 28), 'int')
        # Getting the type of 'tfin' (line 140)
        tfin_292976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'tfin', False)
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___292977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), tfin_292976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_292978 = invoke(stypy.reporting.localization.Localization(__file__, 140, 23), getitem___292977, int_292975)
        
        
        # Obtaining the type of the subscript
        int_292979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'int')
        # Getting the type of 'tfin' (line 140)
        tfin_292980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'tfin', False)
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___292981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 32), tfin_292980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_292982 = invoke(stypy.reporting.localization.Localization(__file__, 140, 32), getitem___292981, int_292979)
        
        # Processing the call keyword arguments (line 140)
        kwargs_292983 = {}
        # Getting the type of 'tf2zpk' (line 140)
        tf2zpk_292974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'tf2zpk', False)
        # Calling tf2zpk(args, kwargs) (line 140)
        tf2zpk_call_result_292984 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), tf2zpk_292974, *[subscript_call_result_292978, subscript_call_result_292982], **kwargs_292983)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_292985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        float_292986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 44), tuple_292985, float_292986)
        
        # Applying the binary operator '+' (line 140)
        result_add_292987 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 16), '+', tf2zpk_call_result_292984, tuple_292985)
        
        # Assigning a type to the variable 'zpkin' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'zpkin', result_add_292987)
        
        # Assigning a Call to a Tuple (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_292988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to dstep(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'zpkin' (line 141)
        zpkin_292990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'zpkin', False)
        # Processing the call keyword arguments (line 141)
        int_292991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'int')
        keyword_292992 = int_292991
        kwargs_292993 = {'n': keyword_292992}
        # Getting the type of 'dstep' (line 141)
        dstep_292989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 141)
        dstep_call_result_292994 = invoke(stypy.reporting.localization.Localization(__file__, 141, 21), dstep_292989, *[zpkin_292990], **kwargs_292993)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___292995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), dstep_call_result_292994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_292996 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___292995, int_292988)
        
        # Assigning a type to the variable 'tuple_var_assignment_292187' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_292187', subscript_call_result_292996)
        
        # Assigning a Subscript to a Name (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_292997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to dstep(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'zpkin' (line 141)
        zpkin_292999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'zpkin', False)
        # Processing the call keyword arguments (line 141)
        int_293000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'int')
        keyword_293001 = int_293000
        kwargs_293002 = {'n': keyword_293001}
        # Getting the type of 'dstep' (line 141)
        dstep_292998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 'dstep', False)
        # Calling dstep(args, kwargs) (line 141)
        dstep_call_result_293003 = invoke(stypy.reporting.localization.Localization(__file__, 141, 21), dstep_292998, *[zpkin_292999], **kwargs_293002)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___293004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), dstep_call_result_293003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_293005 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___293004, int_292997)
        
        # Assigning a type to the variable 'tuple_var_assignment_292188' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_292188', subscript_call_result_293005)
        
        # Assigning a Name to a Name (line 141):
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_292187' (line 141)
        tuple_var_assignment_292187_293006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_292187')
        # Assigning a type to the variable 'tout' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tout', tuple_var_assignment_292187_293006)
        
        # Assigning a Name to a Name (line 141):
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_292188' (line 141)
        tuple_var_assignment_292188_293007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_292188')
        # Assigning a type to the variable 'yout' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 14), 'yout', tuple_var_assignment_292188_293007)
        
        # Call to assert_equal(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to len(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'yout' (line 142)
        yout_293010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'yout', False)
        # Processing the call keyword arguments (line 142)
        kwargs_293011 = {}
        # Getting the type of 'len' (line 142)
        len_293009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'len', False)
        # Calling len(args, kwargs) (line 142)
        len_call_result_293012 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), len_293009, *[yout_293010], **kwargs_293011)
        
        int_293013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 32), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_293014 = {}
        # Getting the type of 'assert_equal' (line 142)
        assert_equal_293008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 142)
        assert_equal_call_result_293015 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), assert_equal_293008, *[len_call_result_293012, int_293013], **kwargs_293014)
        
        
        # Call to assert_array_almost_equal(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Call to flatten(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_293022 = {}
        
        # Obtaining the type of the subscript
        int_293017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 39), 'int')
        # Getting the type of 'yout' (line 143)
        yout_293018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 34), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___293019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 34), yout_293018, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_293020 = invoke(stypy.reporting.localization.Localization(__file__, 143, 34), getitem___293019, int_293017)
        
        # Obtaining the member 'flatten' of a type (line 143)
        flatten_293021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 34), subscript_call_result_293020, 'flatten')
        # Calling flatten(args, kwargs) (line 143)
        flatten_call_result_293023 = invoke(stypy.reporting.localization.Localization(__file__, 143, 34), flatten_293021, *[], **kwargs_293022)
        
        # Getting the type of 'yout_tfstep' (line 143)
        yout_tfstep_293024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 53), 'yout_tfstep', False)
        # Processing the call keyword arguments (line 143)
        kwargs_293025 = {}
        # Getting the type of 'assert_array_almost_equal' (line 143)
        assert_array_almost_equal_293016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 143)
        assert_array_almost_equal_call_result_293026 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert_array_almost_equal_293016, *[flatten_call_result_293023, yout_tfstep_293024], **kwargs_293025)
        
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to lti(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_293028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        int_293029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), list_293028, int_293029)
        
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_293030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        int_293031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 26), list_293030, int_293031)
        # Adding element type (line 146)
        int_293032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 26), list_293030, int_293032)
        
        # Processing the call keyword arguments (line 146)
        kwargs_293033 = {}
        # Getting the type of 'lti' (line 146)
        lti_293027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 17), 'lti', False)
        # Calling lti(args, kwargs) (line 146)
        lti_call_result_293034 = invoke(stypy.reporting.localization.Localization(__file__, 146, 17), lti_293027, *[list_293028, list_293030], **kwargs_293033)
        
        # Assigning a type to the variable 'system' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'system', lti_call_result_293034)
        
        # Call to assert_raises(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'AttributeError' (line 147)
        AttributeError_293036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'AttributeError', False)
        # Getting the type of 'dstep' (line 147)
        dstep_293037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'dstep', False)
        # Getting the type of 'system' (line 147)
        system_293038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'system', False)
        # Processing the call keyword arguments (line 147)
        kwargs_293039 = {}
        # Getting the type of 'assert_raises' (line 147)
        assert_raises_293035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 147)
        assert_raises_call_result_293040 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assert_raises_293035, *[AttributeError_293036, dstep_293037, system_293038], **kwargs_293039)
        
        
        # ################# End of 'test_dstep(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dstep' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_293041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dstep'
        return stypy_return_type_293041


    @norecursion
    def test_dimpulse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dimpulse'
        module_type_store = module_type_store.open_function_context('test_dimpulse', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dimpulse')
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dimpulse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dimpulse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dimpulse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dimpulse(...)' code ##################

        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to asarray(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_293044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_293045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        float_293046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 24), list_293045, float_293046)
        # Adding element type (line 151)
        float_293047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 24), list_293045, float_293047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 23), list_293044, list_293045)
        # Adding element type (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_293048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        float_293049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 36), list_293048, float_293049)
        # Adding element type (line 151)
        float_293050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 36), list_293048, float_293050)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 23), list_293044, list_293048)
        
        # Processing the call keyword arguments (line 151)
        kwargs_293051 = {}
        # Getting the type of 'np' (line 151)
        np_293042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 151)
        asarray_293043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), np_293042, 'asarray')
        # Calling asarray(args, kwargs) (line 151)
        asarray_call_result_293052 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), asarray_293043, *[list_293044], **kwargs_293051)
        
        # Assigning a type to the variable 'a' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'a', asarray_call_result_293052)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to asarray(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_293055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_293056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        float_293057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), list_293056, float_293057)
        # Adding element type (line 152)
        float_293058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), list_293056, float_293058)
        # Adding element type (line 152)
        float_293059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 24), list_293056, float_293059)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 23), list_293055, list_293056)
        # Adding element type (line 152)
        
        # Obtaining an instance of the builtin type 'list' (line 152)
        list_293060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 152)
        # Adding element type (line 152)
        float_293061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 42), list_293060, float_293061)
        # Adding element type (line 152)
        float_293062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 42), list_293060, float_293062)
        # Adding element type (line 152)
        float_293063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 42), list_293060, float_293063)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 23), list_293055, list_293060)
        
        # Processing the call keyword arguments (line 152)
        kwargs_293064 = {}
        # Getting the type of 'np' (line 152)
        np_293053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 152)
        asarray_293054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), np_293053, 'asarray')
        # Calling asarray(args, kwargs) (line 152)
        asarray_call_result_293065 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), asarray_293054, *[list_293055], **kwargs_293064)
        
        # Assigning a type to the variable 'b' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'b', asarray_call_result_293065)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to asarray(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_293068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        # Adding element type (line 153)
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_293069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        # Adding element type (line 153)
        float_293070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 24), list_293069, float_293070)
        # Adding element type (line 153)
        float_293071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 24), list_293069, float_293071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 23), list_293068, list_293069)
        
        # Processing the call keyword arguments (line 153)
        kwargs_293072 = {}
        # Getting the type of 'np' (line 153)
        np_293066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 153)
        asarray_293067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 12), np_293066, 'asarray')
        # Calling asarray(args, kwargs) (line 153)
        asarray_call_result_293073 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), asarray_293067, *[list_293068], **kwargs_293072)
        
        # Assigning a type to the variable 'c' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'c', asarray_call_result_293073)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to asarray(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_293076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_293077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        float_293078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_293077, float_293078)
        # Adding element type (line 154)
        float_293079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_293077, float_293079)
        # Adding element type (line 154)
        float_293080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), list_293077, float_293080)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 23), list_293076, list_293077)
        
        # Processing the call keyword arguments (line 154)
        kwargs_293081 = {}
        # Getting the type of 'np' (line 154)
        np_293074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 154)
        asarray_293075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), np_293074, 'asarray')
        # Calling asarray(args, kwargs) (line 154)
        asarray_call_result_293082 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), asarray_293075, *[list_293076], **kwargs_293081)
        
        # Assigning a type to the variable 'd' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'd', asarray_call_result_293082)
        
        # Assigning a Num to a Name (line 155):
        
        # Assigning a Num to a Name (line 155):
        
        # Assigning a Num to a Name (line 155):
        float_293083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 13), 'float')
        # Assigning a type to the variable 'dt' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'dt', float_293083)
        
        # Assigning a Tuple to a Name (line 159):
        
        # Assigning a Tuple to a Name (line 159):
        
        # Assigning a Tuple to a Name (line 159):
        
        # Obtaining an instance of the builtin type 'tuple' (line 159)
        tuple_293084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 159)
        # Adding element type (line 159)
        
        # Call to asarray(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_293087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        float_293088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293088)
        # Adding element type (line 159)
        float_293089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293089)
        # Adding element type (line 159)
        float_293090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293090)
        # Adding element type (line 159)
        float_293091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293091)
        # Adding element type (line 159)
        float_293092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293092)
        # Adding element type (line 159)
        float_293093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293093)
        # Adding element type (line 159)
        float_293094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293094)
        # Adding element type (line 159)
        float_293095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293095)
        # Adding element type (line 159)
        float_293096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293096)
        # Adding element type (line 159)
        float_293097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 37), list_293087, float_293097)
        
        # Processing the call keyword arguments (line 159)
        kwargs_293098 = {}
        # Getting the type of 'np' (line 159)
        np_293085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'np', False)
        # Obtaining the member 'asarray' of a type (line 159)
        asarray_293086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 26), np_293085, 'asarray')
        # Calling asarray(args, kwargs) (line 159)
        asarray_call_result_293099 = invoke(stypy.reporting.localization.Localization(__file__, 159, 26), asarray_293086, *[list_293087], **kwargs_293098)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 26), tuple_293084, asarray_call_result_293099)
        # Adding element type (line 159)
        
        # Call to asarray(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_293102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        float_293103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293103)
        # Adding element type (line 162)
        float_293104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293104)
        # Adding element type (line 162)
        float_293105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293105)
        # Adding element type (line 162)
        float_293106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293106)
        # Adding element type (line 162)
        float_293107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293107)
        # Adding element type (line 162)
        float_293108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293108)
        # Adding element type (line 162)
        float_293109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293109)
        # Adding element type (line 162)
        float_293110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293110)
        # Adding element type (line 162)
        float_293111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293111)
        # Adding element type (line 162)
        float_293112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), list_293102, float_293112)
        
        # Processing the call keyword arguments (line 162)
        kwargs_293113 = {}
        # Getting the type of 'np' (line 162)
        np_293100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'np', False)
        # Obtaining the member 'asarray' of a type (line 162)
        asarray_293101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 26), np_293100, 'asarray')
        # Calling asarray(args, kwargs) (line 162)
        asarray_call_result_293114 = invoke(stypy.reporting.localization.Localization(__file__, 162, 26), asarray_293101, *[list_293102], **kwargs_293113)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 26), tuple_293084, asarray_call_result_293114)
        # Adding element type (line 159)
        
        # Call to asarray(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Obtaining an instance of the builtin type 'list' (line 165)
        list_293117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 165)
        # Adding element type (line 165)
        float_293118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293118)
        # Adding element type (line 165)
        float_293119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293119)
        # Adding element type (line 165)
        float_293120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293120)
        # Adding element type (line 165)
        float_293121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293121)
        # Adding element type (line 165)
        float_293122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293122)
        # Adding element type (line 165)
        float_293123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293123)
        # Adding element type (line 165)
        float_293124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293124)
        # Adding element type (line 165)
        float_293125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293125)
        # Adding element type (line 165)
        float_293126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293126)
        # Adding element type (line 165)
        float_293127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 37), list_293117, float_293127)
        
        # Processing the call keyword arguments (line 165)
        kwargs_293128 = {}
        # Getting the type of 'np' (line 165)
        np_293115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'np', False)
        # Obtaining the member 'asarray' of a type (line 165)
        asarray_293116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 26), np_293115, 'asarray')
        # Calling asarray(args, kwargs) (line 165)
        asarray_call_result_293129 = invoke(stypy.reporting.localization.Localization(__file__, 165, 26), asarray_293116, *[list_293117], **kwargs_293128)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 26), tuple_293084, asarray_call_result_293129)
        
        # Assigning a type to the variable 'yout_imp_truth' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'yout_imp_truth', tuple_293084)
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Subscript to a Name (line 169):
        
        # Assigning a Subscript to a Name (line 169):
        
        # Obtaining the type of the subscript
        int_293130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
        
        # Call to dimpulse(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_293132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'a' (line 169)
        a_293133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293132, a_293133)
        # Adding element type (line 169)
        # Getting the type of 'b' (line 169)
        b_293134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293132, b_293134)
        # Adding element type (line 169)
        # Getting the type of 'c' (line 169)
        c_293135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293132, c_293135)
        # Adding element type (line 169)
        # Getting the type of 'd' (line 169)
        d_293136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 40), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293132, d_293136)
        # Adding element type (line 169)
        # Getting the type of 'dt' (line 169)
        dt_293137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293132, dt_293137)
        
        # Processing the call keyword arguments (line 169)
        int_293138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'int')
        keyword_293139 = int_293138
        kwargs_293140 = {'n': keyword_293139}
        # Getting the type of 'dimpulse' (line 169)
        dimpulse_293131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 169)
        dimpulse_call_result_293141 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), dimpulse_293131, *[tuple_293132], **kwargs_293140)
        
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___293142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dimpulse_call_result_293141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_293143 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___293142, int_293130)
        
        # Assigning a type to the variable 'tuple_var_assignment_292189' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_292189', subscript_call_result_293143)
        
        # Assigning a Subscript to a Name (line 169):
        
        # Assigning a Subscript to a Name (line 169):
        
        # Obtaining the type of the subscript
        int_293144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
        
        # Call to dimpulse(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Obtaining an instance of the builtin type 'tuple' (line 169)
        tuple_293146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 169)
        # Adding element type (line 169)
        # Getting the type of 'a' (line 169)
        a_293147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293146, a_293147)
        # Adding element type (line 169)
        # Getting the type of 'b' (line 169)
        b_293148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293146, b_293148)
        # Adding element type (line 169)
        # Getting the type of 'c' (line 169)
        c_293149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293146, c_293149)
        # Adding element type (line 169)
        # Getting the type of 'd' (line 169)
        d_293150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 40), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293146, d_293150)
        # Adding element type (line 169)
        # Getting the type of 'dt' (line 169)
        dt_293151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'dt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 31), tuple_293146, dt_293151)
        
        # Processing the call keyword arguments (line 169)
        int_293152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 50), 'int')
        keyword_293153 = int_293152
        kwargs_293154 = {'n': keyword_293153}
        # Getting the type of 'dimpulse' (line 169)
        dimpulse_293145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 169)
        dimpulse_call_result_293155 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), dimpulse_293145, *[tuple_293146], **kwargs_293154)
        
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___293156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), dimpulse_call_result_293155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_293157 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___293156, int_293144)
        
        # Assigning a type to the variable 'tuple_var_assignment_292190' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_292190', subscript_call_result_293157)
        
        # Assigning a Name to a Name (line 169):
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'tuple_var_assignment_292189' (line 169)
        tuple_var_assignment_292189_293158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_292189')
        # Assigning a type to the variable 'tout' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tout', tuple_var_assignment_292189_293158)
        
        # Assigning a Name to a Name (line 169):
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'tuple_var_assignment_292190' (line 169)
        tuple_var_assignment_292190_293159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_292190')
        # Assigning a type to the variable 'yout' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'yout', tuple_var_assignment_292190_293159)
        
        # Call to assert_equal(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to len(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'yout' (line 171)
        yout_293162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'yout', False)
        # Processing the call keyword arguments (line 171)
        kwargs_293163 = {}
        # Getting the type of 'len' (line 171)
        len_293161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 21), 'len', False)
        # Calling len(args, kwargs) (line 171)
        len_call_result_293164 = invoke(stypy.reporting.localization.Localization(__file__, 171, 21), len_293161, *[yout_293162], **kwargs_293163)
        
        int_293165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 32), 'int')
        # Processing the call keyword arguments (line 171)
        kwargs_293166 = {}
        # Getting the type of 'assert_equal' (line 171)
        assert_equal_293160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 171)
        assert_equal_call_result_293167 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_equal_293160, *[len_call_result_293164, int_293165], **kwargs_293166)
        
        
        
        # Call to range(...): (line 173)
        # Processing the call arguments (line 173)
        int_293169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'int')
        
        # Call to len(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'yout' (line 173)
        yout_293171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 30), 'yout', False)
        # Processing the call keyword arguments (line 173)
        kwargs_293172 = {}
        # Getting the type of 'len' (line 173)
        len_293170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'len', False)
        # Calling len(args, kwargs) (line 173)
        len_call_result_293173 = invoke(stypy.reporting.localization.Localization(__file__, 173, 26), len_293170, *[yout_293171], **kwargs_293172)
        
        # Processing the call keyword arguments (line 173)
        kwargs_293174 = {}
        # Getting the type of 'range' (line 173)
        range_293168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'range', False)
        # Calling range(args, kwargs) (line 173)
        range_call_result_293175 = invoke(stypy.reporting.localization.Localization(__file__, 173, 17), range_293168, *[int_293169, len_call_result_293173], **kwargs_293174)
        
        # Testing the type of a for loop iterable (line 173)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 8), range_call_result_293175)
        # Getting the type of the for loop variable (line 173)
        for_loop_var_293176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 8), range_call_result_293175)
        # Assigning a type to the variable 'i' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'i', for_loop_var_293176)
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_equal(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining the type of the subscript
        int_293178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 39), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 174)
        i_293179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'i', False)
        # Getting the type of 'yout' (line 174)
        yout_293180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___293181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), yout_293180, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_293182 = invoke(stypy.reporting.localization.Localization(__file__, 174, 25), getitem___293181, i_293179)
        
        # Obtaining the member 'shape' of a type (line 174)
        shape_293183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), subscript_call_result_293182, 'shape')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___293184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), shape_293183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_293185 = invoke(stypy.reporting.localization.Localization(__file__, 174, 25), getitem___293184, int_293178)
        
        int_293186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 43), 'int')
        # Processing the call keyword arguments (line 174)
        kwargs_293187 = {}
        # Getting the type of 'assert_equal' (line 174)
        assert_equal_293177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 174)
        assert_equal_call_result_293188 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), assert_equal_293177, *[subscript_call_result_293185, int_293186], **kwargs_293187)
        
        
        # Call to assert_array_almost_equal(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to flatten(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_293195 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 175)
        i_293190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'i', False)
        # Getting the type of 'yout' (line 175)
        yout_293191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___293192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 38), yout_293191, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_293193 = invoke(stypy.reporting.localization.Localization(__file__, 175, 38), getitem___293192, i_293190)
        
        # Obtaining the member 'flatten' of a type (line 175)
        flatten_293194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 38), subscript_call_result_293193, 'flatten')
        # Calling flatten(args, kwargs) (line 175)
        flatten_call_result_293196 = invoke(stypy.reporting.localization.Localization(__file__, 175, 38), flatten_293194, *[], **kwargs_293195)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 175)
        i_293197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 72), 'i', False)
        # Getting the type of 'yout_imp_truth' (line 175)
        yout_imp_truth_293198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 57), 'yout_imp_truth', False)
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___293199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 57), yout_imp_truth_293198, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_293200 = invoke(stypy.reporting.localization.Localization(__file__, 175, 57), getitem___293199, i_293197)
        
        # Processing the call keyword arguments (line 175)
        kwargs_293201 = {}
        # Getting the type of 'assert_array_almost_equal' (line 175)
        assert_array_almost_equal_293189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 175)
        assert_array_almost_equal_call_result_293202 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), assert_array_almost_equal_293189, *[flatten_call_result_293196, subscript_call_result_293200], **kwargs_293201)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 178):
        
        # Assigning a Tuple to a Name (line 178):
        
        # Assigning a Tuple to a Name (line 178):
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_293203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_293204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        float_293205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 17), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), list_293204, float_293205)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), tuple_293203, list_293204)
        # Adding element type (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_293206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        float_293207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), list_293206, float_293207)
        # Adding element type (line 178)
        float_293208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), list_293206, float_293208)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), tuple_293203, list_293206)
        # Adding element type (line 178)
        float_293209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), tuple_293203, float_293209)
        
        # Assigning a type to the variable 'tfin' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'tfin', tuple_293203)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to asarray(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_293212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        float_293213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 36), list_293212, float_293213)
        # Adding element type (line 179)
        float_293214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 36), list_293212, float_293214)
        # Adding element type (line 179)
        float_293215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 36), list_293212, float_293215)
        
        # Processing the call keyword arguments (line 179)
        kwargs_293216 = {}
        # Getting the type of 'np' (line 179)
        np_293210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 'np', False)
        # Obtaining the member 'asarray' of a type (line 179)
        asarray_293211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 25), np_293210, 'asarray')
        # Calling asarray(args, kwargs) (line 179)
        asarray_call_result_293217 = invoke(stypy.reporting.localization.Localization(__file__, 179, 25), asarray_293211, *[list_293212], **kwargs_293216)
        
        # Assigning a type to the variable 'yout_tfimpulse' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'yout_tfimpulse', asarray_call_result_293217)
        
        # Assigning a Call to a Tuple (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_293218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to dimpulse(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'tfin' (line 180)
        tfin_293220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'tfin', False)
        # Processing the call keyword arguments (line 180)
        int_293221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 38), 'int')
        keyword_293222 = int_293221
        kwargs_293223 = {'n': keyword_293222}
        # Getting the type of 'dimpulse' (line 180)
        dimpulse_293219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 180)
        dimpulse_call_result_293224 = invoke(stypy.reporting.localization.Localization(__file__, 180, 21), dimpulse_293219, *[tfin_293220], **kwargs_293223)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___293225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), dimpulse_call_result_293224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_293226 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___293225, int_293218)
        
        # Assigning a type to the variable 'tuple_var_assignment_292191' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_292191', subscript_call_result_293226)
        
        # Assigning a Subscript to a Name (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        int_293227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 8), 'int')
        
        # Call to dimpulse(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'tfin' (line 180)
        tfin_293229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'tfin', False)
        # Processing the call keyword arguments (line 180)
        int_293230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 38), 'int')
        keyword_293231 = int_293230
        kwargs_293232 = {'n': keyword_293231}
        # Getting the type of 'dimpulse' (line 180)
        dimpulse_293228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 180)
        dimpulse_call_result_293233 = invoke(stypy.reporting.localization.Localization(__file__, 180, 21), dimpulse_293228, *[tfin_293229], **kwargs_293232)
        
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___293234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), dimpulse_call_result_293233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_293235 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), getitem___293234, int_293227)
        
        # Assigning a type to the variable 'tuple_var_assignment_292192' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_292192', subscript_call_result_293235)
        
        # Assigning a Name to a Name (line 180):
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_292191' (line 180)
        tuple_var_assignment_292191_293236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_292191')
        # Assigning a type to the variable 'tout' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tout', tuple_var_assignment_292191_293236)
        
        # Assigning a Name to a Name (line 180):
        
        # Assigning a Name to a Name (line 180):
        # Getting the type of 'tuple_var_assignment_292192' (line 180)
        tuple_var_assignment_292192_293237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'tuple_var_assignment_292192')
        # Assigning a type to the variable 'yout' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'yout', tuple_var_assignment_292192_293237)
        
        # Call to assert_equal(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to len(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'yout' (line 181)
        yout_293240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 'yout', False)
        # Processing the call keyword arguments (line 181)
        kwargs_293241 = {}
        # Getting the type of 'len' (line 181)
        len_293239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'len', False)
        # Calling len(args, kwargs) (line 181)
        len_call_result_293242 = invoke(stypy.reporting.localization.Localization(__file__, 181, 21), len_293239, *[yout_293240], **kwargs_293241)
        
        int_293243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 32), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_293244 = {}
        # Getting the type of 'assert_equal' (line 181)
        assert_equal_293238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 181)
        assert_equal_call_result_293245 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_equal_293238, *[len_call_result_293242, int_293243], **kwargs_293244)
        
        
        # Call to assert_array_almost_equal(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Call to flatten(...): (line 182)
        # Processing the call keyword arguments (line 182)
        kwargs_293252 = {}
        
        # Obtaining the type of the subscript
        int_293247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'int')
        # Getting the type of 'yout' (line 182)
        yout_293248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___293249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), yout_293248, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_293250 = invoke(stypy.reporting.localization.Localization(__file__, 182, 34), getitem___293249, int_293247)
        
        # Obtaining the member 'flatten' of a type (line 182)
        flatten_293251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), subscript_call_result_293250, 'flatten')
        # Calling flatten(args, kwargs) (line 182)
        flatten_call_result_293253 = invoke(stypy.reporting.localization.Localization(__file__, 182, 34), flatten_293251, *[], **kwargs_293252)
        
        # Getting the type of 'yout_tfimpulse' (line 182)
        yout_tfimpulse_293254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'yout_tfimpulse', False)
        # Processing the call keyword arguments (line 182)
        kwargs_293255 = {}
        # Getting the type of 'assert_array_almost_equal' (line 182)
        assert_array_almost_equal_293246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 182)
        assert_array_almost_equal_call_result_293256 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_array_almost_equal_293246, *[flatten_call_result_293253, yout_tfimpulse_293254], **kwargs_293255)
        
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        
        # Call to tf2zpk(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Obtaining the type of the subscript
        int_293258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'int')
        # Getting the type of 'tfin' (line 184)
        tfin_293259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'tfin', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___293260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 23), tfin_293259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_293261 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), getitem___293260, int_293258)
        
        
        # Obtaining the type of the subscript
        int_293262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 37), 'int')
        # Getting the type of 'tfin' (line 184)
        tfin_293263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 32), 'tfin', False)
        # Obtaining the member '__getitem__' of a type (line 184)
        getitem___293264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 32), tfin_293263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 184)
        subscript_call_result_293265 = invoke(stypy.reporting.localization.Localization(__file__, 184, 32), getitem___293264, int_293262)
        
        # Processing the call keyword arguments (line 184)
        kwargs_293266 = {}
        # Getting the type of 'tf2zpk' (line 184)
        tf2zpk_293257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'tf2zpk', False)
        # Calling tf2zpk(args, kwargs) (line 184)
        tf2zpk_call_result_293267 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), tf2zpk_293257, *[subscript_call_result_293261, subscript_call_result_293265], **kwargs_293266)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_293268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        float_293269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 44), tuple_293268, float_293269)
        
        # Applying the binary operator '+' (line 184)
        result_add_293270 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 16), '+', tf2zpk_call_result_293267, tuple_293268)
        
        # Assigning a type to the variable 'zpkin' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'zpkin', result_add_293270)
        
        # Assigning a Call to a Tuple (line 185):
        
        # Assigning a Subscript to a Name (line 185):
        
        # Assigning a Subscript to a Name (line 185):
        
        # Obtaining the type of the subscript
        int_293271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 8), 'int')
        
        # Call to dimpulse(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'zpkin' (line 185)
        zpkin_293273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'zpkin', False)
        # Processing the call keyword arguments (line 185)
        int_293274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'int')
        keyword_293275 = int_293274
        kwargs_293276 = {'n': keyword_293275}
        # Getting the type of 'dimpulse' (line 185)
        dimpulse_293272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 185)
        dimpulse_call_result_293277 = invoke(stypy.reporting.localization.Localization(__file__, 185, 21), dimpulse_293272, *[zpkin_293273], **kwargs_293276)
        
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___293278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), dimpulse_call_result_293277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_293279 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), getitem___293278, int_293271)
        
        # Assigning a type to the variable 'tuple_var_assignment_292193' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tuple_var_assignment_292193', subscript_call_result_293279)
        
        # Assigning a Subscript to a Name (line 185):
        
        # Assigning a Subscript to a Name (line 185):
        
        # Obtaining the type of the subscript
        int_293280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 8), 'int')
        
        # Call to dimpulse(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'zpkin' (line 185)
        zpkin_293282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'zpkin', False)
        # Processing the call keyword arguments (line 185)
        int_293283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'int')
        keyword_293284 = int_293283
        kwargs_293285 = {'n': keyword_293284}
        # Getting the type of 'dimpulse' (line 185)
        dimpulse_293281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 185)
        dimpulse_call_result_293286 = invoke(stypy.reporting.localization.Localization(__file__, 185, 21), dimpulse_293281, *[zpkin_293282], **kwargs_293285)
        
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___293287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), dimpulse_call_result_293286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_293288 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), getitem___293287, int_293280)
        
        # Assigning a type to the variable 'tuple_var_assignment_292194' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tuple_var_assignment_292194', subscript_call_result_293288)
        
        # Assigning a Name to a Name (line 185):
        
        # Assigning a Name to a Name (line 185):
        # Getting the type of 'tuple_var_assignment_292193' (line 185)
        tuple_var_assignment_292193_293289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tuple_var_assignment_292193')
        # Assigning a type to the variable 'tout' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tout', tuple_var_assignment_292193_293289)
        
        # Assigning a Name to a Name (line 185):
        
        # Assigning a Name to a Name (line 185):
        # Getting the type of 'tuple_var_assignment_292194' (line 185)
        tuple_var_assignment_292194_293290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'tuple_var_assignment_292194')
        # Assigning a type to the variable 'yout' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 14), 'yout', tuple_var_assignment_292194_293290)
        
        # Call to assert_equal(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to len(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'yout' (line 186)
        yout_293293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'yout', False)
        # Processing the call keyword arguments (line 186)
        kwargs_293294 = {}
        # Getting the type of 'len' (line 186)
        len_293292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'len', False)
        # Calling len(args, kwargs) (line 186)
        len_call_result_293295 = invoke(stypy.reporting.localization.Localization(__file__, 186, 21), len_293292, *[yout_293293], **kwargs_293294)
        
        int_293296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 32), 'int')
        # Processing the call keyword arguments (line 186)
        kwargs_293297 = {}
        # Getting the type of 'assert_equal' (line 186)
        assert_equal_293291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 186)
        assert_equal_call_result_293298 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert_equal_293291, *[len_call_result_293295, int_293296], **kwargs_293297)
        
        
        # Call to assert_array_almost_equal(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to flatten(...): (line 187)
        # Processing the call keyword arguments (line 187)
        kwargs_293305 = {}
        
        # Obtaining the type of the subscript
        int_293300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 39), 'int')
        # Getting the type of 'yout' (line 187)
        yout_293301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), 'yout', False)
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___293302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), yout_293301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_293303 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), getitem___293302, int_293300)
        
        # Obtaining the member 'flatten' of a type (line 187)
        flatten_293304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 34), subscript_call_result_293303, 'flatten')
        # Calling flatten(args, kwargs) (line 187)
        flatten_call_result_293306 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), flatten_293304, *[], **kwargs_293305)
        
        # Getting the type of 'yout_tfimpulse' (line 187)
        yout_tfimpulse_293307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 53), 'yout_tfimpulse', False)
        # Processing the call keyword arguments (line 187)
        kwargs_293308 = {}
        # Getting the type of 'assert_array_almost_equal' (line 187)
        assert_array_almost_equal_293299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 187)
        assert_array_almost_equal_call_result_293309 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_array_almost_equal_293299, *[flatten_call_result_293306, yout_tfimpulse_293307], **kwargs_293308)
        
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to lti(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_293311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        int_293312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 21), list_293311, int_293312)
        
        
        # Obtaining an instance of the builtin type 'list' (line 190)
        list_293313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 190)
        # Adding element type (line 190)
        int_293314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 26), list_293313, int_293314)
        # Adding element type (line 190)
        int_293315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 26), list_293313, int_293315)
        
        # Processing the call keyword arguments (line 190)
        kwargs_293316 = {}
        # Getting the type of 'lti' (line 190)
        lti_293310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'lti', False)
        # Calling lti(args, kwargs) (line 190)
        lti_call_result_293317 = invoke(stypy.reporting.localization.Localization(__file__, 190, 17), lti_293310, *[list_293311, list_293313], **kwargs_293316)
        
        # Assigning a type to the variable 'system' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'system', lti_call_result_293317)
        
        # Call to assert_raises(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'AttributeError' (line 191)
        AttributeError_293319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'AttributeError', False)
        # Getting the type of 'dimpulse' (line 191)
        dimpulse_293320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 38), 'dimpulse', False)
        # Getting the type of 'system' (line 191)
        system_293321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'system', False)
        # Processing the call keyword arguments (line 191)
        kwargs_293322 = {}
        # Getting the type of 'assert_raises' (line 191)
        assert_raises_293318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 191)
        assert_raises_call_result_293323 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assert_raises_293318, *[AttributeError_293319, dimpulse_293320, system_293321], **kwargs_293322)
        
        
        # ################# End of 'test_dimpulse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dimpulse' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_293324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293324)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dimpulse'
        return stypy_return_type_293324


    @norecursion
    def test_dlsim_trivial(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlsim_trivial'
        module_type_store = module_type_store.open_function_context('test_dlsim_trivial', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dlsim_trivial')
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dlsim_trivial.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dlsim_trivial', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlsim_trivial', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlsim_trivial(...)' code ##################

        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to array(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_293327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_293328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        float_293329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 22), list_293328, float_293329)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 21), list_293327, list_293328)
        
        # Processing the call keyword arguments (line 194)
        kwargs_293330 = {}
        # Getting the type of 'np' (line 194)
        np_293325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 194)
        array_293326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), np_293325, 'array')
        # Calling array(args, kwargs) (line 194)
        array_call_result_293331 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), array_293326, *[list_293327], **kwargs_293330)
        
        # Assigning a type to the variable 'a' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'a', array_call_result_293331)
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to array(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_293334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_293335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        float_293336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 22), list_293335, float_293336)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 21), list_293334, list_293335)
        
        # Processing the call keyword arguments (line 195)
        kwargs_293337 = {}
        # Getting the type of 'np' (line 195)
        np_293332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 195)
        array_293333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), np_293332, 'array')
        # Calling array(args, kwargs) (line 195)
        array_call_result_293338 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), array_293333, *[list_293334], **kwargs_293337)
        
        # Assigning a type to the variable 'b' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'b', array_call_result_293338)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to array(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_293341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_293342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        float_293343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 22), list_293342, float_293343)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 21), list_293341, list_293342)
        
        # Processing the call keyword arguments (line 196)
        kwargs_293344 = {}
        # Getting the type of 'np' (line 196)
        np_293339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 196)
        array_293340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), np_293339, 'array')
        # Calling array(args, kwargs) (line 196)
        array_call_result_293345 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), array_293340, *[list_293341], **kwargs_293344)
        
        # Assigning a type to the variable 'c' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'c', array_call_result_293345)
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to array(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_293348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_293349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        float_293350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 22), list_293349, float_293350)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_293348, list_293349)
        
        # Processing the call keyword arguments (line 197)
        kwargs_293351 = {}
        # Getting the type of 'np' (line 197)
        np_293346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 197)
        array_293347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), np_293346, 'array')
        # Calling array(args, kwargs) (line 197)
        array_call_result_293352 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), array_293347, *[list_293348], **kwargs_293351)
        
        # Assigning a type to the variable 'd' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'd', array_call_result_293352)
        
        # Assigning a Num to a Name (line 198):
        
        # Assigning a Num to a Name (line 198):
        
        # Assigning a Num to a Name (line 198):
        int_293353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 12), 'int')
        # Assigning a type to the variable 'n' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'n', int_293353)
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Assigning a Call to a Name (line 199):
        
        # Call to reshape(...): (line 199)
        # Processing the call arguments (line 199)
        int_293360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 32), 'int')
        int_293361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'int')
        # Processing the call keyword arguments (line 199)
        kwargs_293362 = {}
        
        # Call to zeros(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'n' (line 199)
        n_293356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'n', False)
        # Processing the call keyword arguments (line 199)
        kwargs_293357 = {}
        # Getting the type of 'np' (line 199)
        np_293354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 199)
        zeros_293355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), np_293354, 'zeros')
        # Calling zeros(args, kwargs) (line 199)
        zeros_call_result_293358 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), zeros_293355, *[n_293356], **kwargs_293357)
        
        # Obtaining the member 'reshape' of a type (line 199)
        reshape_293359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), zeros_call_result_293358, 'reshape')
        # Calling reshape(args, kwargs) (line 199)
        reshape_call_result_293363 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), reshape_293359, *[int_293360, int_293361], **kwargs_293362)
        
        # Assigning a type to the variable 'u' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'u', reshape_call_result_293363)
        
        # Assigning a Call to a Tuple (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_293364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to dlsim(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_293366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        # Getting the type of 'a' (line 200)
        a_293367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293366, a_293367)
        # Adding element type (line 200)
        # Getting the type of 'b' (line 200)
        b_293368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293366, b_293368)
        # Adding element type (line 200)
        # Getting the type of 'c' (line 200)
        c_293369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293366, c_293369)
        # Adding element type (line 200)
        # Getting the type of 'd' (line 200)
        d_293370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293366, d_293370)
        # Adding element type (line 200)
        int_293371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293366, int_293371)
        
        # Getting the type of 'u' (line 200)
        u_293372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'u', False)
        # Processing the call keyword arguments (line 200)
        kwargs_293373 = {}
        # Getting the type of 'dlsim' (line 200)
        dlsim_293365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 200)
        dlsim_call_result_293374 = invoke(stypy.reporting.localization.Localization(__file__, 200, 27), dlsim_293365, *[tuple_293366, u_293372], **kwargs_293373)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___293375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), dlsim_call_result_293374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_293376 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___293375, int_293364)
        
        # Assigning a type to the variable 'tuple_var_assignment_292195' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292195', subscript_call_result_293376)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_293377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to dlsim(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_293379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        # Getting the type of 'a' (line 200)
        a_293380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293379, a_293380)
        # Adding element type (line 200)
        # Getting the type of 'b' (line 200)
        b_293381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293379, b_293381)
        # Adding element type (line 200)
        # Getting the type of 'c' (line 200)
        c_293382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293379, c_293382)
        # Adding element type (line 200)
        # Getting the type of 'd' (line 200)
        d_293383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293379, d_293383)
        # Adding element type (line 200)
        int_293384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293379, int_293384)
        
        # Getting the type of 'u' (line 200)
        u_293385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'u', False)
        # Processing the call keyword arguments (line 200)
        kwargs_293386 = {}
        # Getting the type of 'dlsim' (line 200)
        dlsim_293378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 200)
        dlsim_call_result_293387 = invoke(stypy.reporting.localization.Localization(__file__, 200, 27), dlsim_293378, *[tuple_293379, u_293385], **kwargs_293386)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___293388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), dlsim_call_result_293387, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_293389 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___293388, int_293377)
        
        # Assigning a type to the variable 'tuple_var_assignment_292196' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292196', subscript_call_result_293389)
        
        # Assigning a Subscript to a Name (line 200):
        
        # Assigning a Subscript to a Name (line 200):
        
        # Obtaining the type of the subscript
        int_293390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 8), 'int')
        
        # Call to dlsim(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'tuple' (line 200)
        tuple_293392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 200)
        # Adding element type (line 200)
        # Getting the type of 'a' (line 200)
        a_293393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293392, a_293393)
        # Adding element type (line 200)
        # Getting the type of 'b' (line 200)
        b_293394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293392, b_293394)
        # Adding element type (line 200)
        # Getting the type of 'c' (line 200)
        c_293395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293392, c_293395)
        # Adding element type (line 200)
        # Getting the type of 'd' (line 200)
        d_293396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293392, d_293396)
        # Adding element type (line 200)
        int_293397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 34), tuple_293392, int_293397)
        
        # Getting the type of 'u' (line 200)
        u_293398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 50), 'u', False)
        # Processing the call keyword arguments (line 200)
        kwargs_293399 = {}
        # Getting the type of 'dlsim' (line 200)
        dlsim_293391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 200)
        dlsim_call_result_293400 = invoke(stypy.reporting.localization.Localization(__file__, 200, 27), dlsim_293391, *[tuple_293392, u_293398], **kwargs_293399)
        
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___293401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), dlsim_call_result_293400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_293402 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), getitem___293401, int_293390)
        
        # Assigning a type to the variable 'tuple_var_assignment_292197' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292197', subscript_call_result_293402)
        
        # Assigning a Name to a Name (line 200):
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_292195' (line 200)
        tuple_var_assignment_292195_293403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292195')
        # Assigning a type to the variable 'tout' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tout', tuple_var_assignment_292195_293403)
        
        # Assigning a Name to a Name (line 200):
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_292196' (line 200)
        tuple_var_assignment_292196_293404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292196')
        # Assigning a type to the variable 'yout' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'yout', tuple_var_assignment_292196_293404)
        
        # Assigning a Name to a Name (line 200):
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'tuple_var_assignment_292197' (line 200)
        tuple_var_assignment_292197_293405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'tuple_var_assignment_292197')
        # Assigning a type to the variable 'xout' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'xout', tuple_var_assignment_292197_293405)
        
        # Call to assert_array_equal(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'tout' (line 201)
        tout_293407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'tout', False)
        
        # Call to arange(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to float(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'n' (line 201)
        n_293411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 49), 'n', False)
        # Processing the call keyword arguments (line 201)
        kwargs_293412 = {}
        # Getting the type of 'float' (line 201)
        float_293410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'float', False)
        # Calling float(args, kwargs) (line 201)
        float_call_result_293413 = invoke(stypy.reporting.localization.Localization(__file__, 201, 43), float_293410, *[n_293411], **kwargs_293412)
        
        # Processing the call keyword arguments (line 201)
        kwargs_293414 = {}
        # Getting the type of 'np' (line 201)
        np_293408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'np', False)
        # Obtaining the member 'arange' of a type (line 201)
        arange_293409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 33), np_293408, 'arange')
        # Calling arange(args, kwargs) (line 201)
        arange_call_result_293415 = invoke(stypy.reporting.localization.Localization(__file__, 201, 33), arange_293409, *[float_call_result_293413], **kwargs_293414)
        
        # Processing the call keyword arguments (line 201)
        kwargs_293416 = {}
        # Getting the type of 'assert_array_equal' (line 201)
        assert_array_equal_293406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 201)
        assert_array_equal_call_result_293417 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assert_array_equal_293406, *[tout_293407, arange_call_result_293415], **kwargs_293416)
        
        
        # Call to assert_array_equal(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'yout' (line 202)
        yout_293419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'yout', False)
        
        # Call to zeros(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining an instance of the builtin type 'tuple' (line 202)
        tuple_293422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 202)
        # Adding element type (line 202)
        # Getting the type of 'n' (line 202)
        n_293423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 43), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 43), tuple_293422, n_293423)
        # Adding element type (line 202)
        int_293424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 43), tuple_293422, int_293424)
        
        # Processing the call keyword arguments (line 202)
        kwargs_293425 = {}
        # Getting the type of 'np' (line 202)
        np_293420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 33), 'np', False)
        # Obtaining the member 'zeros' of a type (line 202)
        zeros_293421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 33), np_293420, 'zeros')
        # Calling zeros(args, kwargs) (line 202)
        zeros_call_result_293426 = invoke(stypy.reporting.localization.Localization(__file__, 202, 33), zeros_293421, *[tuple_293422], **kwargs_293425)
        
        # Processing the call keyword arguments (line 202)
        kwargs_293427 = {}
        # Getting the type of 'assert_array_equal' (line 202)
        assert_array_equal_293418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 202)
        assert_array_equal_call_result_293428 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_array_equal_293418, *[yout_293419, zeros_call_result_293426], **kwargs_293427)
        
        
        # Call to assert_array_equal(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'xout' (line 203)
        xout_293430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'xout', False)
        
        # Call to zeros(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_293433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        # Getting the type of 'n' (line 203)
        n_293434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 43), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 43), tuple_293433, n_293434)
        # Adding element type (line 203)
        int_293435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 43), tuple_293433, int_293435)
        
        # Processing the call keyword arguments (line 203)
        kwargs_293436 = {}
        # Getting the type of 'np' (line 203)
        np_293431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 33), 'np', False)
        # Obtaining the member 'zeros' of a type (line 203)
        zeros_293432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 33), np_293431, 'zeros')
        # Calling zeros(args, kwargs) (line 203)
        zeros_call_result_293437 = invoke(stypy.reporting.localization.Localization(__file__, 203, 33), zeros_293432, *[tuple_293433], **kwargs_293436)
        
        # Processing the call keyword arguments (line 203)
        kwargs_293438 = {}
        # Getting the type of 'assert_array_equal' (line 203)
        assert_array_equal_293429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 203)
        assert_array_equal_call_result_293439 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), assert_array_equal_293429, *[xout_293430, zeros_call_result_293437], **kwargs_293438)
        
        
        # ################# End of 'test_dlsim_trivial(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlsim_trivial' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_293440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlsim_trivial'
        return stypy_return_type_293440


    @norecursion
    def test_dlsim_simple1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlsim_simple1d'
        module_type_store = module_type_store.open_function_context('test_dlsim_simple1d', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dlsim_simple1d')
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dlsim_simple1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dlsim_simple1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlsim_simple1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlsim_simple1d(...)' code ##################

        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to array(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_293443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        
        # Obtaining an instance of the builtin type 'list' (line 206)
        list_293444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 206)
        # Adding element type (line 206)
        float_293445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 22), list_293444, float_293445)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 21), list_293443, list_293444)
        
        # Processing the call keyword arguments (line 206)
        kwargs_293446 = {}
        # Getting the type of 'np' (line 206)
        np_293441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 206)
        array_293442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), np_293441, 'array')
        # Calling array(args, kwargs) (line 206)
        array_call_result_293447 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), array_293442, *[list_293443], **kwargs_293446)
        
        # Assigning a type to the variable 'a' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'a', array_call_result_293447)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to array(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_293450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_293451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        float_293452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 22), list_293451, float_293452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 21), list_293450, list_293451)
        
        # Processing the call keyword arguments (line 207)
        kwargs_293453 = {}
        # Getting the type of 'np' (line 207)
        np_293448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 207)
        array_293449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), np_293448, 'array')
        # Calling array(args, kwargs) (line 207)
        array_call_result_293454 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), array_293449, *[list_293450], **kwargs_293453)
        
        # Assigning a type to the variable 'b' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'b', array_call_result_293454)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to array(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_293457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_293458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        float_293459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 22), list_293458, float_293459)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 21), list_293457, list_293458)
        
        # Processing the call keyword arguments (line 208)
        kwargs_293460 = {}
        # Getting the type of 'np' (line 208)
        np_293455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 208)
        array_293456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), np_293455, 'array')
        # Calling array(args, kwargs) (line 208)
        array_call_result_293461 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), array_293456, *[list_293457], **kwargs_293460)
        
        # Assigning a type to the variable 'c' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'c', array_call_result_293461)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to array(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_293464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_293465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        float_293466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 22), list_293465, float_293466)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 21), list_293464, list_293465)
        
        # Processing the call keyword arguments (line 209)
        kwargs_293467 = {}
        # Getting the type of 'np' (line 209)
        np_293462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 209)
        array_293463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), np_293462, 'array')
        # Calling array(args, kwargs) (line 209)
        array_call_result_293468 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), array_293463, *[list_293464], **kwargs_293467)
        
        # Assigning a type to the variable 'd' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'd', array_call_result_293468)
        
        # Assigning a Num to a Name (line 210):
        
        # Assigning a Num to a Name (line 210):
        
        # Assigning a Num to a Name (line 210):
        int_293469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'int')
        # Assigning a type to the variable 'n' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'n', int_293469)
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to reshape(...): (line 211)
        # Processing the call arguments (line 211)
        int_293476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'int')
        int_293477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 36), 'int')
        # Processing the call keyword arguments (line 211)
        kwargs_293478 = {}
        
        # Call to zeros(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'n' (line 211)
        n_293472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'n', False)
        # Processing the call keyword arguments (line 211)
        kwargs_293473 = {}
        # Getting the type of 'np' (line 211)
        np_293470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 211)
        zeros_293471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), np_293470, 'zeros')
        # Calling zeros(args, kwargs) (line 211)
        zeros_call_result_293474 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), zeros_293471, *[n_293472], **kwargs_293473)
        
        # Obtaining the member 'reshape' of a type (line 211)
        reshape_293475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), zeros_call_result_293474, 'reshape')
        # Calling reshape(args, kwargs) (line 211)
        reshape_call_result_293479 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), reshape_293475, *[int_293476, int_293477], **kwargs_293478)
        
        # Assigning a type to the variable 'u' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'u', reshape_call_result_293479)
        
        # Assigning a Call to a Tuple (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_293480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to dlsim(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_293482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'a' (line 212)
        a_293483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293482, a_293483)
        # Adding element type (line 212)
        # Getting the type of 'b' (line 212)
        b_293484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293482, b_293484)
        # Adding element type (line 212)
        # Getting the type of 'c' (line 212)
        c_293485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293482, c_293485)
        # Adding element type (line 212)
        # Getting the type of 'd' (line 212)
        d_293486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293482, d_293486)
        # Adding element type (line 212)
        int_293487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293482, int_293487)
        
        # Getting the type of 'u' (line 212)
        u_293488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 50), 'u', False)
        # Processing the call keyword arguments (line 212)
        int_293489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 56), 'int')
        keyword_293490 = int_293489
        kwargs_293491 = {'x0': keyword_293490}
        # Getting the type of 'dlsim' (line 212)
        dlsim_293481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 212)
        dlsim_call_result_293492 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), dlsim_293481, *[tuple_293482, u_293488], **kwargs_293491)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___293493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), dlsim_call_result_293492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_293494 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___293493, int_293480)
        
        # Assigning a type to the variable 'tuple_var_assignment_292198' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292198', subscript_call_result_293494)
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_293495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to dlsim(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_293497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'a' (line 212)
        a_293498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293497, a_293498)
        # Adding element type (line 212)
        # Getting the type of 'b' (line 212)
        b_293499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293497, b_293499)
        # Adding element type (line 212)
        # Getting the type of 'c' (line 212)
        c_293500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293497, c_293500)
        # Adding element type (line 212)
        # Getting the type of 'd' (line 212)
        d_293501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293497, d_293501)
        # Adding element type (line 212)
        int_293502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293497, int_293502)
        
        # Getting the type of 'u' (line 212)
        u_293503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 50), 'u', False)
        # Processing the call keyword arguments (line 212)
        int_293504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 56), 'int')
        keyword_293505 = int_293504
        kwargs_293506 = {'x0': keyword_293505}
        # Getting the type of 'dlsim' (line 212)
        dlsim_293496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 212)
        dlsim_call_result_293507 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), dlsim_293496, *[tuple_293497, u_293503], **kwargs_293506)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___293508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), dlsim_call_result_293507, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_293509 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___293508, int_293495)
        
        # Assigning a type to the variable 'tuple_var_assignment_292199' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292199', subscript_call_result_293509)
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_293510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to dlsim(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_293512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'a' (line 212)
        a_293513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293512, a_293513)
        # Adding element type (line 212)
        # Getting the type of 'b' (line 212)
        b_293514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293512, b_293514)
        # Adding element type (line 212)
        # Getting the type of 'c' (line 212)
        c_293515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293512, c_293515)
        # Adding element type (line 212)
        # Getting the type of 'd' (line 212)
        d_293516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293512, d_293516)
        # Adding element type (line 212)
        int_293517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 34), tuple_293512, int_293517)
        
        # Getting the type of 'u' (line 212)
        u_293518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 50), 'u', False)
        # Processing the call keyword arguments (line 212)
        int_293519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 56), 'int')
        keyword_293520 = int_293519
        kwargs_293521 = {'x0': keyword_293520}
        # Getting the type of 'dlsim' (line 212)
        dlsim_293511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 212)
        dlsim_call_result_293522 = invoke(stypy.reporting.localization.Localization(__file__, 212, 27), dlsim_293511, *[tuple_293512, u_293518], **kwargs_293521)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___293523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), dlsim_call_result_293522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_293524 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___293523, int_293510)
        
        # Assigning a type to the variable 'tuple_var_assignment_292200' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292200', subscript_call_result_293524)
        
        # Assigning a Name to a Name (line 212):
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_292198' (line 212)
        tuple_var_assignment_292198_293525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292198')
        # Assigning a type to the variable 'tout' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tout', tuple_var_assignment_292198_293525)
        
        # Assigning a Name to a Name (line 212):
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_292199' (line 212)
        tuple_var_assignment_292199_293526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292199')
        # Assigning a type to the variable 'yout' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'yout', tuple_var_assignment_292199_293526)
        
        # Assigning a Name to a Name (line 212):
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_292200' (line 212)
        tuple_var_assignment_292200_293527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_292200')
        # Assigning a type to the variable 'xout' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'xout', tuple_var_assignment_292200_293527)
        
        # Call to assert_array_equal(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'tout' (line 213)
        tout_293529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'tout', False)
        
        # Call to arange(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to float(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'n' (line 213)
        n_293533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 49), 'n', False)
        # Processing the call keyword arguments (line 213)
        kwargs_293534 = {}
        # Getting the type of 'float' (line 213)
        float_293532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 43), 'float', False)
        # Calling float(args, kwargs) (line 213)
        float_call_result_293535 = invoke(stypy.reporting.localization.Localization(__file__, 213, 43), float_293532, *[n_293533], **kwargs_293534)
        
        # Processing the call keyword arguments (line 213)
        kwargs_293536 = {}
        # Getting the type of 'np' (line 213)
        np_293530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'np', False)
        # Obtaining the member 'arange' of a type (line 213)
        arange_293531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 33), np_293530, 'arange')
        # Calling arange(args, kwargs) (line 213)
        arange_call_result_293537 = invoke(stypy.reporting.localization.Localization(__file__, 213, 33), arange_293531, *[float_call_result_293535], **kwargs_293536)
        
        # Processing the call keyword arguments (line 213)
        kwargs_293538 = {}
        # Getting the type of 'assert_array_equal' (line 213)
        assert_array_equal_293528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 213)
        assert_array_equal_call_result_293539 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_array_equal_293528, *[tout_293529, arange_call_result_293537], **kwargs_293538)
        
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to reshape(...): (line 214)
        # Processing the call arguments (line 214)
        int_293551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 56), 'int')
        int_293552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 60), 'int')
        # Processing the call keyword arguments (line 214)
        kwargs_293553 = {}
        float_293540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 20), 'float')
        
        # Call to arange(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to float(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'n' (line 214)
        n_293544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'n', False)
        # Processing the call keyword arguments (line 214)
        kwargs_293545 = {}
        # Getting the type of 'float' (line 214)
        float_293543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 37), 'float', False)
        # Calling float(args, kwargs) (line 214)
        float_call_result_293546 = invoke(stypy.reporting.localization.Localization(__file__, 214, 37), float_293543, *[n_293544], **kwargs_293545)
        
        # Processing the call keyword arguments (line 214)
        kwargs_293547 = {}
        # Getting the type of 'np' (line 214)
        np_293541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'np', False)
        # Obtaining the member 'arange' of a type (line 214)
        arange_293542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 27), np_293541, 'arange')
        # Calling arange(args, kwargs) (line 214)
        arange_call_result_293548 = invoke(stypy.reporting.localization.Localization(__file__, 214, 27), arange_293542, *[float_call_result_293546], **kwargs_293547)
        
        # Applying the binary operator '**' (line 214)
        result_pow_293549 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 20), '**', float_293540, arange_call_result_293548)
        
        # Obtaining the member 'reshape' of a type (line 214)
        reshape_293550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), result_pow_293549, 'reshape')
        # Calling reshape(args, kwargs) (line 214)
        reshape_call_result_293554 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), reshape_293550, *[int_293551, int_293552], **kwargs_293553)
        
        # Assigning a type to the variable 'expected' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'expected', reshape_call_result_293554)
        
        # Call to assert_array_equal(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'yout' (line 215)
        yout_293556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), 'yout', False)
        # Getting the type of 'expected' (line 215)
        expected_293557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'expected', False)
        # Processing the call keyword arguments (line 215)
        kwargs_293558 = {}
        # Getting the type of 'assert_array_equal' (line 215)
        assert_array_equal_293555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 215)
        assert_array_equal_call_result_293559 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_array_equal_293555, *[yout_293556, expected_293557], **kwargs_293558)
        
        
        # Call to assert_array_equal(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'xout' (line 216)
        xout_293561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 27), 'xout', False)
        # Getting the type of 'expected' (line 216)
        expected_293562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 33), 'expected', False)
        # Processing the call keyword arguments (line 216)
        kwargs_293563 = {}
        # Getting the type of 'assert_array_equal' (line 216)
        assert_array_equal_293560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 216)
        assert_array_equal_call_result_293564 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), assert_array_equal_293560, *[xout_293561, expected_293562], **kwargs_293563)
        
        
        # ################# End of 'test_dlsim_simple1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlsim_simple1d' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_293565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlsim_simple1d'
        return stypy_return_type_293565


    @norecursion
    def test_dlsim_simple2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlsim_simple2d'
        module_type_store = module_type_store.open_function_context('test_dlsim_simple2d', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_dlsim_simple2d')
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_dlsim_simple2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_dlsim_simple2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlsim_simple2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlsim_simple2d(...)' code ##################

        
        # Assigning a Num to a Name (line 219):
        
        # Assigning a Num to a Name (line 219):
        
        # Assigning a Num to a Name (line 219):
        float_293566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'float')
        # Assigning a type to the variable 'lambda1' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'lambda1', float_293566)
        
        # Assigning a Num to a Name (line 220):
        
        # Assigning a Num to a Name (line 220):
        
        # Assigning a Num to a Name (line 220):
        float_293567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 18), 'float')
        # Assigning a type to the variable 'lambda2' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'lambda2', float_293567)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to array(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_293570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        # Adding element type (line 221)
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_293571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        # Adding element type (line 221)
        # Getting the type of 'lambda1' (line 221)
        lambda1_293572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'lambda1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 22), list_293571, lambda1_293572)
        # Adding element type (line 221)
        float_293573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 22), list_293571, float_293573)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_293570, list_293571)
        # Adding element type (line 221)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_293574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        float_293575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_293574, float_293575)
        # Adding element type (line 222)
        # Getting the type of 'lambda2' (line 222)
        lambda2_293576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'lambda2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 22), list_293574, lambda2_293576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_293570, list_293574)
        
        # Processing the call keyword arguments (line 221)
        kwargs_293577 = {}
        # Getting the type of 'np' (line 221)
        np_293568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 221)
        array_293569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), np_293568, 'array')
        # Calling array(args, kwargs) (line 221)
        array_call_result_293578 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), array_293569, *[list_293570], **kwargs_293577)
        
        # Assigning a type to the variable 'a' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'a', array_call_result_293578)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to array(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_293581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 223)
        list_293582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 223)
        # Adding element type (line 223)
        float_293583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 22), list_293582, float_293583)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_293581, list_293582)
        # Adding element type (line 223)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_293584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        float_293585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 22), list_293584, float_293585)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 21), list_293581, list_293584)
        
        # Processing the call keyword arguments (line 223)
        kwargs_293586 = {}
        # Getting the type of 'np' (line 223)
        np_293579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 223)
        array_293580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), np_293579, 'array')
        # Calling array(args, kwargs) (line 223)
        array_call_result_293587 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), array_293580, *[list_293581], **kwargs_293586)
        
        # Assigning a type to the variable 'b' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'b', array_call_result_293587)
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to array(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_293590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_293591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        float_293592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 22), list_293591, float_293592)
        # Adding element type (line 225)
        float_293593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 22), list_293591, float_293593)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 21), list_293590, list_293591)
        # Adding element type (line 225)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_293594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        float_293595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 22), list_293594, float_293595)
        # Adding element type (line 226)
        float_293596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 22), list_293594, float_293596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 21), list_293590, list_293594)
        
        # Processing the call keyword arguments (line 225)
        kwargs_293597 = {}
        # Getting the type of 'np' (line 225)
        np_293588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 225)
        array_293589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), np_293588, 'array')
        # Calling array(args, kwargs) (line 225)
        array_call_result_293598 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), array_293589, *[list_293590], **kwargs_293597)
        
        # Assigning a type to the variable 'c' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'c', array_call_result_293598)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to array(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_293601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_293602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        float_293603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_293602, float_293603)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_293601, list_293602)
        # Adding element type (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 228)
        list_293604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 228)
        # Adding element type (line 228)
        float_293605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 22), list_293604, float_293605)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 21), list_293601, list_293604)
        
        # Processing the call keyword arguments (line 227)
        kwargs_293606 = {}
        # Getting the type of 'np' (line 227)
        np_293599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 227)
        array_293600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), np_293599, 'array')
        # Calling array(args, kwargs) (line 227)
        array_call_result_293607 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), array_293600, *[list_293601], **kwargs_293606)
        
        # Assigning a type to the variable 'd' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'd', array_call_result_293607)
        
        # Assigning a Num to a Name (line 229):
        
        # Assigning a Num to a Name (line 229):
        
        # Assigning a Num to a Name (line 229):
        int_293608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 12), 'int')
        # Assigning a type to the variable 'n' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'n', int_293608)
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to reshape(...): (line 230)
        # Processing the call arguments (line 230)
        int_293615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 32), 'int')
        int_293616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 36), 'int')
        # Processing the call keyword arguments (line 230)
        kwargs_293617 = {}
        
        # Call to zeros(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'n' (line 230)
        n_293611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'n', False)
        # Processing the call keyword arguments (line 230)
        kwargs_293612 = {}
        # Getting the type of 'np' (line 230)
        np_293609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 230)
        zeros_293610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), np_293609, 'zeros')
        # Calling zeros(args, kwargs) (line 230)
        zeros_call_result_293613 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), zeros_293610, *[n_293611], **kwargs_293612)
        
        # Obtaining the member 'reshape' of a type (line 230)
        reshape_293614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), zeros_call_result_293613, 'reshape')
        # Calling reshape(args, kwargs) (line 230)
        reshape_call_result_293618 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), reshape_293614, *[int_293615, int_293616], **kwargs_293617)
        
        # Assigning a type to the variable 'u' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'u', reshape_call_result_293618)
        
        # Assigning a Call to a Tuple (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_293619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to dlsim(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_293621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        # Getting the type of 'a' (line 231)
        a_293622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293621, a_293622)
        # Adding element type (line 231)
        # Getting the type of 'b' (line 231)
        b_293623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293621, b_293623)
        # Adding element type (line 231)
        # Getting the type of 'c' (line 231)
        c_293624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293621, c_293624)
        # Adding element type (line 231)
        # Getting the type of 'd' (line 231)
        d_293625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293621, d_293625)
        # Adding element type (line 231)
        int_293626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293621, int_293626)
        
        # Getting the type of 'u' (line 231)
        u_293627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 50), 'u', False)
        # Processing the call keyword arguments (line 231)
        int_293628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 56), 'int')
        keyword_293629 = int_293628
        kwargs_293630 = {'x0': keyword_293629}
        # Getting the type of 'dlsim' (line 231)
        dlsim_293620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 231)
        dlsim_call_result_293631 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), dlsim_293620, *[tuple_293621, u_293627], **kwargs_293630)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___293632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), dlsim_call_result_293631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_293633 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___293632, int_293619)
        
        # Assigning a type to the variable 'tuple_var_assignment_292201' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292201', subscript_call_result_293633)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_293634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to dlsim(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_293636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        # Getting the type of 'a' (line 231)
        a_293637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293636, a_293637)
        # Adding element type (line 231)
        # Getting the type of 'b' (line 231)
        b_293638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293636, b_293638)
        # Adding element type (line 231)
        # Getting the type of 'c' (line 231)
        c_293639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293636, c_293639)
        # Adding element type (line 231)
        # Getting the type of 'd' (line 231)
        d_293640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293636, d_293640)
        # Adding element type (line 231)
        int_293641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293636, int_293641)
        
        # Getting the type of 'u' (line 231)
        u_293642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 50), 'u', False)
        # Processing the call keyword arguments (line 231)
        int_293643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 56), 'int')
        keyword_293644 = int_293643
        kwargs_293645 = {'x0': keyword_293644}
        # Getting the type of 'dlsim' (line 231)
        dlsim_293635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 231)
        dlsim_call_result_293646 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), dlsim_293635, *[tuple_293636, u_293642], **kwargs_293645)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___293647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), dlsim_call_result_293646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_293648 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___293647, int_293634)
        
        # Assigning a type to the variable 'tuple_var_assignment_292202' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292202', subscript_call_result_293648)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_293649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to dlsim(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Obtaining an instance of the builtin type 'tuple' (line 231)
        tuple_293651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 231)
        # Adding element type (line 231)
        # Getting the type of 'a' (line 231)
        a_293652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 34), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293651, a_293652)
        # Adding element type (line 231)
        # Getting the type of 'b' (line 231)
        b_293653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 37), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293651, b_293653)
        # Adding element type (line 231)
        # Getting the type of 'c' (line 231)
        c_293654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293651, c_293654)
        # Adding element type (line 231)
        # Getting the type of 'd' (line 231)
        d_293655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 43), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293651, d_293655)
        # Adding element type (line 231)
        int_293656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 34), tuple_293651, int_293656)
        
        # Getting the type of 'u' (line 231)
        u_293657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 50), 'u', False)
        # Processing the call keyword arguments (line 231)
        int_293658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 56), 'int')
        keyword_293659 = int_293658
        kwargs_293660 = {'x0': keyword_293659}
        # Getting the type of 'dlsim' (line 231)
        dlsim_293650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 231)
        dlsim_call_result_293661 = invoke(stypy.reporting.localization.Localization(__file__, 231, 27), dlsim_293650, *[tuple_293651, u_293657], **kwargs_293660)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___293662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), dlsim_call_result_293661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_293663 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___293662, int_293649)
        
        # Assigning a type to the variable 'tuple_var_assignment_292203' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292203', subscript_call_result_293663)
        
        # Assigning a Name to a Name (line 231):
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_292201' (line 231)
        tuple_var_assignment_292201_293664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292201')
        # Assigning a type to the variable 'tout' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tout', tuple_var_assignment_292201_293664)
        
        # Assigning a Name to a Name (line 231):
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_292202' (line 231)
        tuple_var_assignment_292202_293665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292202')
        # Assigning a type to the variable 'yout' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 14), 'yout', tuple_var_assignment_292202_293665)
        
        # Assigning a Name to a Name (line 231):
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_292203' (line 231)
        tuple_var_assignment_292203_293666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_292203')
        # Assigning a type to the variable 'xout' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'xout', tuple_var_assignment_292203_293666)
        
        # Call to assert_array_equal(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'tout' (line 232)
        tout_293668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'tout', False)
        
        # Call to arange(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to float(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'n' (line 232)
        n_293672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 49), 'n', False)
        # Processing the call keyword arguments (line 232)
        kwargs_293673 = {}
        # Getting the type of 'float' (line 232)
        float_293671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 43), 'float', False)
        # Calling float(args, kwargs) (line 232)
        float_call_result_293674 = invoke(stypy.reporting.localization.Localization(__file__, 232, 43), float_293671, *[n_293672], **kwargs_293673)
        
        # Processing the call keyword arguments (line 232)
        kwargs_293675 = {}
        # Getting the type of 'np' (line 232)
        np_293669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'np', False)
        # Obtaining the member 'arange' of a type (line 232)
        arange_293670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 33), np_293669, 'arange')
        # Calling arange(args, kwargs) (line 232)
        arange_call_result_293676 = invoke(stypy.reporting.localization.Localization(__file__, 232, 33), arange_293670, *[float_call_result_293674], **kwargs_293675)
        
        # Processing the call keyword arguments (line 232)
        kwargs_293677 = {}
        # Getting the type of 'assert_array_equal' (line 232)
        assert_array_equal_293667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 232)
        assert_array_equal_call_result_293678 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), assert_array_equal_293667, *[tout_293668, arange_call_result_293676], **kwargs_293677)
        
        
        # Assigning a BinOp to a Name (line 234):
        
        # Assigning a BinOp to a Name (line 234):
        
        # Assigning a BinOp to a Name (line 234):
        
        # Call to array(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Obtaining an instance of the builtin type 'list' (line 234)
        list_293681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 234)
        # Adding element type (line 234)
        # Getting the type of 'lambda1' (line 234)
        lambda1_293682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 30), 'lambda1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 29), list_293681, lambda1_293682)
        # Adding element type (line 234)
        # Getting the type of 'lambda2' (line 234)
        lambda2_293683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 39), 'lambda2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 29), list_293681, lambda2_293683)
        
        # Processing the call keyword arguments (line 234)
        kwargs_293684 = {}
        # Getting the type of 'np' (line 234)
        np_293679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'np', False)
        # Obtaining the member 'array' of a type (line 234)
        array_293680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 20), np_293679, 'array')
        # Calling array(args, kwargs) (line 234)
        array_call_result_293685 = invoke(stypy.reporting.localization.Localization(__file__, 234, 20), array_293680, *[list_293681], **kwargs_293684)
        
        
        # Call to reshape(...): (line 235)
        # Processing the call arguments (line 235)
        int_293695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 60), 'int')
        int_293696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 64), 'int')
        # Processing the call keyword arguments (line 235)
        kwargs_293697 = {}
        
        # Call to arange(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Call to float(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'n' (line 235)
        n_293689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 48), 'n', False)
        # Processing the call keyword arguments (line 235)
        kwargs_293690 = {}
        # Getting the type of 'float' (line 235)
        float_293688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 42), 'float', False)
        # Calling float(args, kwargs) (line 235)
        float_call_result_293691 = invoke(stypy.reporting.localization.Localization(__file__, 235, 42), float_293688, *[n_293689], **kwargs_293690)
        
        # Processing the call keyword arguments (line 235)
        kwargs_293692 = {}
        # Getting the type of 'np' (line 235)
        np_293686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'np', False)
        # Obtaining the member 'arange' of a type (line 235)
        arange_293687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 32), np_293686, 'arange')
        # Calling arange(args, kwargs) (line 235)
        arange_call_result_293693 = invoke(stypy.reporting.localization.Localization(__file__, 235, 32), arange_293687, *[float_call_result_293691], **kwargs_293692)
        
        # Obtaining the member 'reshape' of a type (line 235)
        reshape_293694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 32), arange_call_result_293693, 'reshape')
        # Calling reshape(args, kwargs) (line 235)
        reshape_call_result_293698 = invoke(stypy.reporting.localization.Localization(__file__, 235, 32), reshape_293694, *[int_293695, int_293696], **kwargs_293697)
        
        # Applying the binary operator '**' (line 234)
        result_pow_293699 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 20), '**', array_call_result_293685, reshape_call_result_293698)
        
        # Assigning a type to the variable 'expected' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'expected', result_pow_293699)
        
        # Call to assert_array_equal(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'yout' (line 236)
        yout_293701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'yout', False)
        # Getting the type of 'expected' (line 236)
        expected_293702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'expected', False)
        # Processing the call keyword arguments (line 236)
        kwargs_293703 = {}
        # Getting the type of 'assert_array_equal' (line 236)
        assert_array_equal_293700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 236)
        assert_array_equal_call_result_293704 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assert_array_equal_293700, *[yout_293701, expected_293702], **kwargs_293703)
        
        
        # Call to assert_array_equal(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'xout' (line 237)
        xout_293706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'xout', False)
        # Getting the type of 'expected' (line 237)
        expected_293707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'expected', False)
        # Processing the call keyword arguments (line 237)
        kwargs_293708 = {}
        # Getting the type of 'assert_array_equal' (line 237)
        assert_array_equal_293705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 237)
        assert_array_equal_call_result_293709 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assert_array_equal_293705, *[xout_293706, expected_293707], **kwargs_293708)
        
        
        # ################# End of 'test_dlsim_simple2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlsim_simple2d' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_293710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlsim_simple2d'
        return stypy_return_type_293710


    @norecursion
    def test_more_step_and_impulse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_more_step_and_impulse'
        module_type_store = module_type_store.open_function_context('test_more_step_and_impulse', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_localization', localization)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_function_name', 'TestDLTI.test_more_step_and_impulse')
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_param_names_list', [])
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDLTI.test_more_step_and_impulse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.test_more_step_and_impulse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_more_step_and_impulse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_more_step_and_impulse(...)' code ##################

        
        # Assigning a Num to a Name (line 240):
        
        # Assigning a Num to a Name (line 240):
        
        # Assigning a Num to a Name (line 240):
        float_293711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 18), 'float')
        # Assigning a type to the variable 'lambda1' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'lambda1', float_293711)
        
        # Assigning a Num to a Name (line 241):
        
        # Assigning a Num to a Name (line 241):
        
        # Assigning a Num to a Name (line 241):
        float_293712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 18), 'float')
        # Assigning a type to the variable 'lambda2' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'lambda2', float_293712)
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to array(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Obtaining an instance of the builtin type 'list' (line 242)
        list_293715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 242)
        # Adding element type (line 242)
        
        # Obtaining an instance of the builtin type 'list' (line 242)
        list_293716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 242)
        # Adding element type (line 242)
        # Getting the type of 'lambda1' (line 242)
        lambda1_293717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'lambda1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 22), list_293716, lambda1_293717)
        # Adding element type (line 242)
        float_293718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 22), list_293716, float_293718)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 21), list_293715, list_293716)
        # Adding element type (line 242)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_293719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        float_293720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 22), list_293719, float_293720)
        # Adding element type (line 243)
        # Getting the type of 'lambda2' (line 243)
        lambda2_293721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'lambda2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 22), list_293719, lambda2_293721)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 21), list_293715, list_293719)
        
        # Processing the call keyword arguments (line 242)
        kwargs_293722 = {}
        # Getting the type of 'np' (line 242)
        np_293713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 242)
        array_293714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), np_293713, 'array')
        # Calling array(args, kwargs) (line 242)
        array_call_result_293723 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), array_293714, *[list_293715], **kwargs_293722)
        
        # Assigning a type to the variable 'a' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'a', array_call_result_293723)
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to array(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_293726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_293727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        float_293728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 22), list_293727, float_293728)
        # Adding element type (line 244)
        float_293729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 22), list_293727, float_293729)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 21), list_293726, list_293727)
        # Adding element type (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_293730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        float_293731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 22), list_293730, float_293731)
        # Adding element type (line 245)
        float_293732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 22), list_293730, float_293732)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 21), list_293726, list_293730)
        
        # Processing the call keyword arguments (line 244)
        kwargs_293733 = {}
        # Getting the type of 'np' (line 244)
        np_293724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 244)
        array_293725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), np_293724, 'array')
        # Calling array(args, kwargs) (line 244)
        array_call_result_293734 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), array_293725, *[list_293726], **kwargs_293733)
        
        # Assigning a type to the variable 'b' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'b', array_call_result_293734)
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to array(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_293737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_293738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        float_293739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 22), list_293738, float_293739)
        # Adding element type (line 246)
        float_293740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 22), list_293738, float_293740)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 21), list_293737, list_293738)
        
        # Processing the call keyword arguments (line 246)
        kwargs_293741 = {}
        # Getting the type of 'np' (line 246)
        np_293735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 246)
        array_293736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), np_293735, 'array')
        # Calling array(args, kwargs) (line 246)
        array_call_result_293742 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), array_293736, *[list_293737], **kwargs_293741)
        
        # Assigning a type to the variable 'c' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'c', array_call_result_293742)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to array(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_293745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_293746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        float_293747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 22), list_293746, float_293747)
        # Adding element type (line 247)
        float_293748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 22), list_293746, float_293748)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 21), list_293745, list_293746)
        
        # Processing the call keyword arguments (line 247)
        kwargs_293749 = {}
        # Getting the type of 'np' (line 247)
        np_293743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 247)
        array_293744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), np_293743, 'array')
        # Calling array(args, kwargs) (line 247)
        array_call_result_293750 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), array_293744, *[list_293745], **kwargs_293749)
        
        # Assigning a type to the variable 'd' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'd', array_call_result_293750)
        
        # Assigning a Num to a Name (line 249):
        
        # Assigning a Num to a Name (line 249):
        
        # Assigning a Num to a Name (line 249):
        int_293751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'int')
        # Assigning a type to the variable 'n' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'n', int_293751)
        
        # Assigning a Call to a Tuple (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_293752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 8), 'int')
        
        # Call to dstep(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_293754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'a' (line 252)
        a_293755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293754, a_293755)
        # Adding element type (line 252)
        # Getting the type of 'b' (line 252)
        b_293756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293754, b_293756)
        # Adding element type (line 252)
        # Getting the type of 'c' (line 252)
        c_293757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293754, c_293757)
        # Adding element type (line 252)
        # Getting the type of 'd' (line 252)
        d_293758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293754, d_293758)
        # Adding element type (line 252)
        int_293759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293754, int_293759)
        
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'n' (line 252)
        n_293760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'n', False)
        keyword_293761 = n_293760
        kwargs_293762 = {'n': keyword_293761}
        # Getting the type of 'dstep' (line 252)
        dstep_293753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'dstep', False)
        # Calling dstep(args, kwargs) (line 252)
        dstep_call_result_293763 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), dstep_293753, *[tuple_293754], **kwargs_293762)
        
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___293764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), dstep_call_result_293763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_293765 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), getitem___293764, int_293752)
        
        # Assigning a type to the variable 'tuple_var_assignment_292204' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_292204', subscript_call_result_293765)
        
        # Assigning a Subscript to a Name (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_293766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 8), 'int')
        
        # Call to dstep(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_293768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'a' (line 252)
        a_293769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 24), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293768, a_293769)
        # Adding element type (line 252)
        # Getting the type of 'b' (line 252)
        b_293770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293768, b_293770)
        # Adding element type (line 252)
        # Getting the type of 'c' (line 252)
        c_293771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 30), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293768, c_293771)
        # Adding element type (line 252)
        # Getting the type of 'd' (line 252)
        d_293772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 33), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293768, d_293772)
        # Adding element type (line 252)
        int_293773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_293768, int_293773)
        
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'n' (line 252)
        n_293774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'n', False)
        keyword_293775 = n_293774
        kwargs_293776 = {'n': keyword_293775}
        # Getting the type of 'dstep' (line 252)
        dstep_293767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'dstep', False)
        # Calling dstep(args, kwargs) (line 252)
        dstep_call_result_293777 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), dstep_293767, *[tuple_293768], **kwargs_293776)
        
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___293778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), dstep_call_result_293777, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_293779 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), getitem___293778, int_293766)
        
        # Assigning a type to the variable 'tuple_var_assignment_292205' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_292205', subscript_call_result_293779)
        
        # Assigning a Name to a Name (line 252):
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_292204' (line 252)
        tuple_var_assignment_292204_293780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_292204')
        # Assigning a type to the variable 'ts' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'ts', tuple_var_assignment_292204_293780)
        
        # Assigning a Name to a Name (line 252):
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_292205' (line 252)
        tuple_var_assignment_292205_293781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_292205')
        # Assigning a type to the variable 'ys' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'ys', tuple_var_assignment_292205_293781)
        
        # Assigning a BinOp to a Name (line 255):
        
        # Assigning a BinOp to a Name (line 255):
        
        # Assigning a BinOp to a Name (line 255):
        float_293782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 16), 'float')
        int_293783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'int')
        # Getting the type of 'lambda1' (line 255)
        lambda1_293784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'lambda1')
        # Applying the binary operator '-' (line 255)
        result_sub_293785 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 23), '-', int_293783, lambda1_293784)
        
        # Applying the binary operator 'div' (line 255)
        result_div_293786 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), 'div', float_293782, result_sub_293785)
        
        float_293787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 40), 'float')
        # Getting the type of 'lambda1' (line 255)
        lambda1_293788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 46), 'lambda1')
        
        # Call to arange(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'n' (line 255)
        n_293791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 67), 'n', False)
        # Processing the call keyword arguments (line 255)
        kwargs_293792 = {}
        # Getting the type of 'np' (line 255)
        np_293789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 57), 'np', False)
        # Obtaining the member 'arange' of a type (line 255)
        arange_293790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 57), np_293789, 'arange')
        # Calling arange(args, kwargs) (line 255)
        arange_call_result_293793 = invoke(stypy.reporting.localization.Localization(__file__, 255, 57), arange_293790, *[n_293791], **kwargs_293792)
        
        # Applying the binary operator '**' (line 255)
        result_pow_293794 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 46), '**', lambda1_293788, arange_call_result_293793)
        
        # Applying the binary operator '-' (line 255)
        result_sub_293795 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 40), '-', float_293787, result_pow_293794)
        
        # Applying the binary operator '*' (line 255)
        result_mul_293796 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 15), '*', result_div_293786, result_sub_293795)
        
        # Assigning a type to the variable 'stp0' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stp0', result_mul_293796)
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        float_293797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 16), 'float')
        int_293798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 23), 'int')
        # Getting the type of 'lambda2' (line 256)
        lambda2_293799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'lambda2')
        # Applying the binary operator '-' (line 256)
        result_sub_293800 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 23), '-', int_293798, lambda2_293799)
        
        # Applying the binary operator 'div' (line 256)
        result_div_293801 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 16), 'div', float_293797, result_sub_293800)
        
        float_293802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'float')
        # Getting the type of 'lambda2' (line 256)
        lambda2_293803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 46), 'lambda2')
        
        # Call to arange(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'n' (line 256)
        n_293806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 67), 'n', False)
        # Processing the call keyword arguments (line 256)
        kwargs_293807 = {}
        # Getting the type of 'np' (line 256)
        np_293804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 57), 'np', False)
        # Obtaining the member 'arange' of a type (line 256)
        arange_293805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 57), np_293804, 'arange')
        # Calling arange(args, kwargs) (line 256)
        arange_call_result_293808 = invoke(stypy.reporting.localization.Localization(__file__, 256, 57), arange_293805, *[n_293806], **kwargs_293807)
        
        # Applying the binary operator '**' (line 256)
        result_pow_293809 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 46), '**', lambda2_293803, arange_call_result_293808)
        
        # Applying the binary operator '-' (line 256)
        result_sub_293810 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 40), '-', float_293802, result_pow_293809)
        
        # Applying the binary operator '*' (line 256)
        result_mul_293811 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 15), '*', result_div_293801, result_sub_293810)
        
        # Assigning a type to the variable 'stp1' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'stp1', result_mul_293811)
        
        # Call to assert_allclose(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining the type of the subscript
        slice_293813 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 258, 24), None, None, None)
        int_293814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 33), 'int')
        
        # Obtaining the type of the subscript
        int_293815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 27), 'int')
        # Getting the type of 'ys' (line 258)
        ys_293816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___293817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 24), ys_293816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_293818 = invoke(stypy.reporting.localization.Localization(__file__, 258, 24), getitem___293817, int_293815)
        
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___293819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 24), subscript_call_result_293818, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_293820 = invoke(stypy.reporting.localization.Localization(__file__, 258, 24), getitem___293819, (slice_293813, int_293814))
        
        # Getting the type of 'stp0' (line 258)
        stp0_293821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'stp0', False)
        # Processing the call keyword arguments (line 258)
        kwargs_293822 = {}
        # Getting the type of 'assert_allclose' (line 258)
        assert_allclose_293812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 258)
        assert_allclose_call_result_293823 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), assert_allclose_293812, *[subscript_call_result_293820, stp0_293821], **kwargs_293822)
        
        
        # Call to assert_allclose(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining the type of the subscript
        slice_293825 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 24), None, None, None)
        int_293826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 33), 'int')
        
        # Obtaining the type of the subscript
        int_293827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
        # Getting the type of 'ys' (line 259)
        ys_293828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'ys', False)
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___293829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), ys_293828, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_293830 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), getitem___293829, int_293827)
        
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___293831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), subscript_call_result_293830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_293832 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), getitem___293831, (slice_293825, int_293826))
        
        # Getting the type of 'stp1' (line 259)
        stp1_293833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'stp1', False)
        # Processing the call keyword arguments (line 259)
        kwargs_293834 = {}
        # Getting the type of 'assert_allclose' (line 259)
        assert_allclose_293824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 259)
        assert_allclose_call_result_293835 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), assert_allclose_293824, *[subscript_call_result_293832, stp1_293833], **kwargs_293834)
        
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to array(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Obtaining an instance of the builtin type 'list' (line 262)
        list_293838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 262)
        # Adding element type (line 262)
        float_293839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 22), list_293838, float_293839)
        # Adding element type (line 262)
        float_293840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 22), list_293838, float_293840)
        
        # Processing the call keyword arguments (line 262)
        kwargs_293841 = {}
        # Getting the type of 'np' (line 262)
        np_293836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 262)
        array_293837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 13), np_293836, 'array')
        # Calling array(args, kwargs) (line 262)
        array_call_result_293842 = invoke(stypy.reporting.localization.Localization(__file__, 262, 13), array_293837, *[list_293838], **kwargs_293841)
        
        # Assigning a type to the variable 'x0' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'x0', array_call_result_293842)
        
        # Assigning a Call to a Tuple (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_293843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 8), 'int')
        
        # Call to dimpulse(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'tuple' (line 263)
        tuple_293845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 263)
        # Adding element type (line 263)
        # Getting the type of 'a' (line 263)
        a_293846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293845, a_293846)
        # Adding element type (line 263)
        # Getting the type of 'b' (line 263)
        b_293847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293845, b_293847)
        # Adding element type (line 263)
        # Getting the type of 'c' (line 263)
        c_293848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293845, c_293848)
        # Adding element type (line 263)
        # Getting the type of 'd' (line 263)
        d_293849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 36), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293845, d_293849)
        # Adding element type (line 263)
        int_293850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293845, int_293850)
        
        # Processing the call keyword arguments (line 263)
        # Getting the type of 'n' (line 263)
        n_293851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'n', False)
        keyword_293852 = n_293851
        # Getting the type of 'x0' (line 263)
        x0_293853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 51), 'x0', False)
        keyword_293854 = x0_293853
        kwargs_293855 = {'x0': keyword_293854, 'n': keyword_293852}
        # Getting the type of 'dimpulse' (line 263)
        dimpulse_293844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 263)
        dimpulse_call_result_293856 = invoke(stypy.reporting.localization.Localization(__file__, 263, 17), dimpulse_293844, *[tuple_293845], **kwargs_293855)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___293857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), dimpulse_call_result_293856, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_293858 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___293857, int_293843)
        
        # Assigning a type to the variable 'tuple_var_assignment_292206' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_292206', subscript_call_result_293858)
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_293859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 8), 'int')
        
        # Call to dimpulse(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'tuple' (line 263)
        tuple_293861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 263)
        # Adding element type (line 263)
        # Getting the type of 'a' (line 263)
        a_293862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 27), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293861, a_293862)
        # Adding element type (line 263)
        # Getting the type of 'b' (line 263)
        b_293863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293861, b_293863)
        # Adding element type (line 263)
        # Getting the type of 'c' (line 263)
        c_293864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293861, c_293864)
        # Adding element type (line 263)
        # Getting the type of 'd' (line 263)
        d_293865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 36), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293861, d_293865)
        # Adding element type (line 263)
        int_293866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), tuple_293861, int_293866)
        
        # Processing the call keyword arguments (line 263)
        # Getting the type of 'n' (line 263)
        n_293867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 45), 'n', False)
        keyword_293868 = n_293867
        # Getting the type of 'x0' (line 263)
        x0_293869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 51), 'x0', False)
        keyword_293870 = x0_293869
        kwargs_293871 = {'x0': keyword_293870, 'n': keyword_293868}
        # Getting the type of 'dimpulse' (line 263)
        dimpulse_293860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 263)
        dimpulse_call_result_293872 = invoke(stypy.reporting.localization.Localization(__file__, 263, 17), dimpulse_293860, *[tuple_293861], **kwargs_293871)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___293873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), dimpulse_call_result_293872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_293874 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___293873, int_293859)
        
        # Assigning a type to the variable 'tuple_var_assignment_292207' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_292207', subscript_call_result_293874)
        
        # Assigning a Name to a Name (line 263):
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_292206' (line 263)
        tuple_var_assignment_292206_293875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_292206')
        # Assigning a type to the variable 'ti' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ti', tuple_var_assignment_292206_293875)
        
        # Assigning a Name to a Name (line 263):
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_292207' (line 263)
        tuple_var_assignment_292207_293876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_292207')
        # Assigning a type to the variable 'yi' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'yi', tuple_var_assignment_292207_293876)
        
        # Assigning a BinOp to a Name (line 266):
        
        # Assigning a BinOp to a Name (line 266):
        
        # Assigning a BinOp to a Name (line 266):
        
        # Call to array(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_293879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        # Getting the type of 'lambda1' (line 266)
        lambda1_293880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'lambda1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 24), list_293879, lambda1_293880)
        # Adding element type (line 266)
        # Getting the type of 'lambda2' (line 266)
        lambda2_293881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'lambda2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 24), list_293879, lambda2_293881)
        
        # Processing the call keyword arguments (line 266)
        kwargs_293882 = {}
        # Getting the type of 'np' (line 266)
        np_293877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 266)
        array_293878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), np_293877, 'array')
        # Calling array(args, kwargs) (line 266)
        array_call_result_293883 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), array_293878, *[list_293879], **kwargs_293882)
        
        
        # Call to reshape(...): (line 267)
        # Processing the call arguments (line 267)
        int_293893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 57), 'int')
        int_293894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 61), 'int')
        # Processing the call keyword arguments (line 267)
        kwargs_293895 = {}
        
        # Call to arange(...): (line 267)
        # Processing the call arguments (line 267)
        int_293886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 38), 'int')
        # Getting the type of 'n' (line 267)
        n_293887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 42), 'n', False)
        int_293888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 46), 'int')
        # Applying the binary operator '+' (line 267)
        result_add_293889 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 42), '+', n_293887, int_293888)
        
        # Processing the call keyword arguments (line 267)
        kwargs_293890 = {}
        # Getting the type of 'np' (line 267)
        np_293884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), 'np', False)
        # Obtaining the member 'arange' of a type (line 267)
        arange_293885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 28), np_293884, 'arange')
        # Calling arange(args, kwargs) (line 267)
        arange_call_result_293891 = invoke(stypy.reporting.localization.Localization(__file__, 267, 28), arange_293885, *[int_293886, result_add_293889], **kwargs_293890)
        
        # Obtaining the member 'reshape' of a type (line 267)
        reshape_293892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 28), arange_call_result_293891, 'reshape')
        # Calling reshape(args, kwargs) (line 267)
        reshape_call_result_293896 = invoke(stypy.reporting.localization.Localization(__file__, 267, 28), reshape_293892, *[int_293893, int_293894], **kwargs_293895)
        
        # Applying the binary operator '**' (line 266)
        result_pow_293897 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 15), '**', array_call_result_293883, reshape_call_result_293896)
        
        # Assigning a type to the variable 'imp' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'imp', result_pow_293897)
        
        # Assigning a Num to a Subscript (line 268):
        
        # Assigning a Num to a Subscript (line 268):
        
        # Assigning a Num to a Subscript (line 268):
        float_293898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 20), 'float')
        # Getting the type of 'imp' (line 268)
        imp_293899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'imp')
        int_293900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 12), 'int')
        slice_293901 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 268, 8), None, None, None)
        # Storing an element on a container (line 268)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 8), imp_293899, ((int_293900, slice_293901), float_293898))
        
        # Assigning a BinOp to a Name (line 270):
        
        # Assigning a BinOp to a Name (line 270):
        
        # Assigning a BinOp to a Name (line 270):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 270)
        n_293902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 18), 'n')
        slice_293903 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 13), None, n_293902, None)
        int_293904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'int')
        # Getting the type of 'imp' (line 270)
        imp_293905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 13), 'imp')
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___293906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 13), imp_293905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_293907 = invoke(stypy.reporting.localization.Localization(__file__, 270, 13), getitem___293906, (slice_293903, int_293904))
        
        
        # Call to dot(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Obtaining the type of the subscript
        int_293910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 37), 'int')
        # Getting the type of 'n' (line 270)
        n_293911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 39), 'n', False)
        int_293912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 43), 'int')
        # Applying the binary operator '+' (line 270)
        result_add_293913 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 39), '+', n_293911, int_293912)
        
        slice_293914 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 33), int_293910, result_add_293913, None)
        slice_293915 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 270, 33), None, None, None)
        # Getting the type of 'imp' (line 270)
        imp_293916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 33), 'imp', False)
        # Obtaining the member '__getitem__' of a type (line 270)
        getitem___293917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 33), imp_293916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 270)
        subscript_call_result_293918 = invoke(stypy.reporting.localization.Localization(__file__, 270, 33), getitem___293917, (slice_293914, slice_293915))
        
        # Getting the type of 'x0' (line 270)
        x0_293919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 50), 'x0', False)
        # Processing the call keyword arguments (line 270)
        kwargs_293920 = {}
        # Getting the type of 'np' (line 270)
        np_293908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'np', False)
        # Obtaining the member 'dot' of a type (line 270)
        dot_293909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 26), np_293908, 'dot')
        # Calling dot(args, kwargs) (line 270)
        dot_call_result_293921 = invoke(stypy.reporting.localization.Localization(__file__, 270, 26), dot_293909, *[subscript_call_result_293918, x0_293919], **kwargs_293920)
        
        # Applying the binary operator '+' (line 270)
        result_add_293922 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 13), '+', subscript_call_result_293907, dot_call_result_293921)
        
        # Assigning a type to the variable 'y0' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'y0', result_add_293922)
        
        # Assigning a BinOp to a Name (line 271):
        
        # Assigning a BinOp to a Name (line 271):
        
        # Assigning a BinOp to a Name (line 271):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 271)
        n_293923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'n')
        slice_293924 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 13), None, n_293923, None)
        int_293925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 21), 'int')
        # Getting the type of 'imp' (line 271)
        imp_293926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'imp')
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___293927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 13), imp_293926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_293928 = invoke(stypy.reporting.localization.Localization(__file__, 271, 13), getitem___293927, (slice_293924, int_293925))
        
        
        # Call to dot(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Obtaining the type of the subscript
        int_293931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 37), 'int')
        # Getting the type of 'n' (line 271)
        n_293932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 39), 'n', False)
        int_293933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 43), 'int')
        # Applying the binary operator '+' (line 271)
        result_add_293934 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 39), '+', n_293932, int_293933)
        
        slice_293935 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 33), int_293931, result_add_293934, None)
        slice_293936 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 271, 33), None, None, None)
        # Getting the type of 'imp' (line 271)
        imp_293937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 33), 'imp', False)
        # Obtaining the member '__getitem__' of a type (line 271)
        getitem___293938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 33), imp_293937, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 271)
        subscript_call_result_293939 = invoke(stypy.reporting.localization.Localization(__file__, 271, 33), getitem___293938, (slice_293935, slice_293936))
        
        # Getting the type of 'x0' (line 271)
        x0_293940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 50), 'x0', False)
        # Processing the call keyword arguments (line 271)
        kwargs_293941 = {}
        # Getting the type of 'np' (line 271)
        np_293929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'np', False)
        # Obtaining the member 'dot' of a type (line 271)
        dot_293930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 26), np_293929, 'dot')
        # Calling dot(args, kwargs) (line 271)
        dot_call_result_293942 = invoke(stypy.reporting.localization.Localization(__file__, 271, 26), dot_293930, *[subscript_call_result_293939, x0_293940], **kwargs_293941)
        
        # Applying the binary operator '+' (line 271)
        result_add_293943 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 13), '+', subscript_call_result_293928, dot_call_result_293942)
        
        # Assigning a type to the variable 'y1' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'y1', result_add_293943)
        
        # Call to assert_allclose(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Obtaining the type of the subscript
        slice_293945 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 273, 24), None, None, None)
        int_293946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'int')
        
        # Obtaining the type of the subscript
        int_293947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'int')
        # Getting the type of 'yi' (line 273)
        yi_293948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'yi', False)
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___293949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), yi_293948, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_293950 = invoke(stypy.reporting.localization.Localization(__file__, 273, 24), getitem___293949, int_293947)
        
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___293951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), subscript_call_result_293950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_293952 = invoke(stypy.reporting.localization.Localization(__file__, 273, 24), getitem___293951, (slice_293945, int_293946))
        
        # Getting the type of 'y0' (line 273)
        y0_293953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 37), 'y0', False)
        # Processing the call keyword arguments (line 273)
        kwargs_293954 = {}
        # Getting the type of 'assert_allclose' (line 273)
        assert_allclose_293944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 273)
        assert_allclose_call_result_293955 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), assert_allclose_293944, *[subscript_call_result_293952, y0_293953], **kwargs_293954)
        
        
        # Call to assert_allclose(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining the type of the subscript
        slice_293957 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 24), None, None, None)
        int_293958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 33), 'int')
        
        # Obtaining the type of the subscript
        int_293959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 27), 'int')
        # Getting the type of 'yi' (line 274)
        yi_293960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'yi', False)
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___293961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), yi_293960, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_293962 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), getitem___293961, int_293959)
        
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___293963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), subscript_call_result_293962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_293964 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), getitem___293963, (slice_293957, int_293958))
        
        # Getting the type of 'y1' (line 274)
        y1_293965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 37), 'y1', False)
        # Processing the call keyword arguments (line 274)
        kwargs_293966 = {}
        # Getting the type of 'assert_allclose' (line 274)
        assert_allclose_293956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 274)
        assert_allclose_call_result_293967 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), assert_allclose_293956, *[subscript_call_result_293964, y1_293965], **kwargs_293966)
        
        
        # Assigning a Tuple to a Name (line 277):
        
        # Assigning a Tuple to a Name (line 277):
        
        # Assigning a Tuple to a Name (line 277):
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_293968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_293969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        float_293970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), list_293969, float_293970)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), tuple_293968, list_293969)
        # Adding element type (line 277)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_293971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        float_293972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 25), list_293971, float_293972)
        # Adding element type (line 277)
        float_293973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 25), list_293971, float_293973)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), tuple_293968, list_293971)
        # Adding element type (line 277)
        float_293974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 18), tuple_293968, float_293974)
        
        # Assigning a type to the variable 'system' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'system', tuple_293968)
        
        # Assigning a Call to a Tuple (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        int_293975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
        
        # Call to dstep(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'system' (line 278)
        system_293977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'system', False)
        # Processing the call keyword arguments (line 278)
        int_293978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 34), 'int')
        keyword_293979 = int_293978
        kwargs_293980 = {'n': keyword_293979}
        # Getting the type of 'dstep' (line 278)
        dstep_293976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'dstep', False)
        # Calling dstep(args, kwargs) (line 278)
        dstep_call_result_293981 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), dstep_293976, *[system_293977], **kwargs_293980)
        
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___293982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), dstep_call_result_293981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_293983 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), getitem___293982, int_293975)
        
        # Assigning a type to the variable 'tuple_var_assignment_292208' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292208', subscript_call_result_293983)
        
        # Assigning a Subscript to a Name (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        int_293984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
        
        # Call to dstep(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'system' (line 278)
        system_293986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'system', False)
        # Processing the call keyword arguments (line 278)
        int_293987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 34), 'int')
        keyword_293988 = int_293987
        kwargs_293989 = {'n': keyword_293988}
        # Getting the type of 'dstep' (line 278)
        dstep_293985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'dstep', False)
        # Calling dstep(args, kwargs) (line 278)
        dstep_call_result_293990 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), dstep_293985, *[system_293986], **kwargs_293989)
        
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___293991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), dstep_call_result_293990, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_293992 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), getitem___293991, int_293984)
        
        # Assigning a type to the variable 'tuple_var_assignment_292209' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292209', subscript_call_result_293992)
        
        # Assigning a Name to a Name (line 278):
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'tuple_var_assignment_292208' (line 278)
        tuple_var_assignment_292208_293993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292208')
        # Assigning a type to the variable 't' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 't', tuple_var_assignment_292208_293993)
        
        # Assigning a Name to a Tuple (line 278):
        
        # Assigning a Subscript to a Name (line 278):
        
        # Obtaining the type of the subscript
        int_293994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
        # Getting the type of 'tuple_var_assignment_292209' (line 278)
        tuple_var_assignment_292209_293995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292209')
        # Obtaining the member '__getitem__' of a type (line 278)
        getitem___293996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), tuple_var_assignment_292209_293995, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 278)
        subscript_call_result_293997 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), getitem___293996, int_293994)
        
        # Assigning a type to the variable 'tuple_var_assignment_292252' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292252', subscript_call_result_293997)
        
        # Assigning a Name to a Name (line 278):
        # Getting the type of 'tuple_var_assignment_292252' (line 278)
        tuple_var_assignment_292252_293998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_292252')
        # Assigning a type to the variable 'y' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'y', tuple_var_assignment_292252_293998)
        
        # Call to assert_allclose(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 't' (line 279)
        t_294000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_294001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_294002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 27), list_294001, int_294002)
        # Adding element type (line 279)
        float_294003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 27), list_294001, float_294003)
        # Adding element type (line 279)
        float_294004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 27), list_294001, float_294004)
        
        # Processing the call keyword arguments (line 279)
        kwargs_294005 = {}
        # Getting the type of 'assert_allclose' (line 279)
        assert_allclose_293999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 279)
        assert_allclose_call_result_294006 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert_allclose_293999, *[t_294000, list_294001], **kwargs_294005)
        
        
        # Call to assert_array_equal(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'y' (line 280)
        y_294008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'y', False)
        # Obtaining the member 'T' of a type (line 280)
        T_294009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 27), y_294008, 'T')
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_294010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_294011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_294012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 33), list_294011, int_294012)
        # Adding element type (line 280)
        float_294013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 33), list_294011, float_294013)
        # Adding element type (line 280)
        float_294014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 33), list_294011, float_294014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 32), list_294010, list_294011)
        
        # Processing the call keyword arguments (line 280)
        kwargs_294015 = {}
        # Getting the type of 'assert_array_equal' (line 280)
        assert_array_equal_294007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 280)
        assert_array_equal_call_result_294016 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assert_array_equal_294007, *[T_294009, list_294010], **kwargs_294015)
        
        
        # Assigning a Call to a Tuple (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        int_294017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 8), 'int')
        
        # Call to dimpulse(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'system' (line 281)
        system_294019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'system', False)
        # Processing the call keyword arguments (line 281)
        int_294020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 37), 'int')
        keyword_294021 = int_294020
        kwargs_294022 = {'n': keyword_294021}
        # Getting the type of 'dimpulse' (line 281)
        dimpulse_294018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 281)
        dimpulse_call_result_294023 = invoke(stypy.reporting.localization.Localization(__file__, 281, 18), dimpulse_294018, *[system_294019], **kwargs_294022)
        
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___294024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), dimpulse_call_result_294023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_294025 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), getitem___294024, int_294017)
        
        # Assigning a type to the variable 'tuple_var_assignment_292210' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292210', subscript_call_result_294025)
        
        # Assigning a Subscript to a Name (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        int_294026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 8), 'int')
        
        # Call to dimpulse(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'system' (line 281)
        system_294028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'system', False)
        # Processing the call keyword arguments (line 281)
        int_294029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 37), 'int')
        keyword_294030 = int_294029
        kwargs_294031 = {'n': keyword_294030}
        # Getting the type of 'dimpulse' (line 281)
        dimpulse_294027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'dimpulse', False)
        # Calling dimpulse(args, kwargs) (line 281)
        dimpulse_call_result_294032 = invoke(stypy.reporting.localization.Localization(__file__, 281, 18), dimpulse_294027, *[system_294028], **kwargs_294031)
        
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___294033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), dimpulse_call_result_294032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_294034 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), getitem___294033, int_294026)
        
        # Assigning a type to the variable 'tuple_var_assignment_292211' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292211', subscript_call_result_294034)
        
        # Assigning a Name to a Name (line 281):
        
        # Assigning a Name to a Name (line 281):
        # Getting the type of 'tuple_var_assignment_292210' (line 281)
        tuple_var_assignment_292210_294035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292210')
        # Assigning a type to the variable 't' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 't', tuple_var_assignment_292210_294035)
        
        # Assigning a Name to a Tuple (line 281):
        
        # Assigning a Subscript to a Name (line 281):
        
        # Obtaining the type of the subscript
        int_294036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 8), 'int')
        # Getting the type of 'tuple_var_assignment_292211' (line 281)
        tuple_var_assignment_292211_294037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292211')
        # Obtaining the member '__getitem__' of a type (line 281)
        getitem___294038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), tuple_var_assignment_292211_294037, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 281)
        subscript_call_result_294039 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), getitem___294038, int_294036)
        
        # Assigning a type to the variable 'tuple_var_assignment_292253' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292253', subscript_call_result_294039)
        
        # Assigning a Name to a Name (line 281):
        # Getting the type of 'tuple_var_assignment_292253' (line 281)
        tuple_var_assignment_292253_294040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'tuple_var_assignment_292253')
        # Assigning a type to the variable 'y' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'y', tuple_var_assignment_292253_294040)
        
        # Call to assert_allclose(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 't' (line 282)
        t_294042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 't', False)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_294043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        int_294044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 27), list_294043, int_294044)
        # Adding element type (line 282)
        float_294045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 27), list_294043, float_294045)
        # Adding element type (line 282)
        float_294046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 27), list_294043, float_294046)
        
        # Processing the call keyword arguments (line 282)
        kwargs_294047 = {}
        # Getting the type of 'assert_allclose' (line 282)
        assert_allclose_294041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 282)
        assert_allclose_call_result_294048 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assert_allclose_294041, *[t_294042, list_294043], **kwargs_294047)
        
        
        # Call to assert_array_equal(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'y' (line 283)
        y_294050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'y', False)
        # Obtaining the member 'T' of a type (line 283)
        T_294051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), y_294050, 'T')
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_294052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_294053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        int_294054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 33), list_294053, int_294054)
        # Adding element type (line 283)
        int_294055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 33), list_294053, int_294055)
        # Adding element type (line 283)
        float_294056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 33), list_294053, float_294056)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 32), list_294052, list_294053)
        
        # Processing the call keyword arguments (line 283)
        kwargs_294057 = {}
        # Getting the type of 'assert_array_equal' (line 283)
        assert_array_equal_294049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 283)
        assert_array_equal_call_result_294058 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert_array_equal_294049, *[T_294051, list_294052], **kwargs_294057)
        
        
        # ################# End of 'test_more_step_and_impulse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_more_step_and_impulse' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_294059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_more_step_and_impulse'
        return stypy_return_type_294059


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDLTI.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDLTI' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'TestDLTI', TestDLTI)
# Declaration of the 'TestDlti' class

class TestDlti(object, ):

    @norecursion
    def test_dlti_instantiation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlti_instantiation'
        module_type_store = module_type_store.open_function_context('test_dlti_instantiation', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_localization', localization)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_function_name', 'TestDlti.test_dlti_instantiation')
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_param_names_list', [])
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDlti.test_dlti_instantiation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDlti.test_dlti_instantiation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlti_instantiation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlti_instantiation(...)' code ##################

        
        # Assigning a Num to a Name (line 290):
        
        # Assigning a Num to a Name (line 290):
        
        # Assigning a Num to a Name (line 290):
        float_294060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 13), 'float')
        # Assigning a type to the variable 'dt' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'dt', float_294060)
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Assigning a Call to a Name (line 292):
        
        # Call to dlti(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_294062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        int_294063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 17), list_294062, int_294063)
        
        
        # Obtaining an instance of the builtin type 'list' (line 292)
        list_294064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 292)
        # Adding element type (line 292)
        int_294065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 22), list_294064, int_294065)
        
        # Processing the call keyword arguments (line 292)
        # Getting the type of 'dt' (line 292)
        dt_294066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'dt', False)
        keyword_294067 = dt_294066
        kwargs_294068 = {'dt': keyword_294067}
        # Getting the type of 'dlti' (line 292)
        dlti_294061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'dlti', False)
        # Calling dlti(args, kwargs) (line 292)
        dlti_call_result_294069 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), dlti_294061, *[list_294062, list_294064], **kwargs_294068)
        
        # Assigning a type to the variable 's' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 's', dlti_call_result_294069)
        
        # Call to assert_(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to isinstance(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 's' (line 293)
        s_294072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 's', False)
        # Getting the type of 'TransferFunction' (line 293)
        TransferFunction_294073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'TransferFunction', False)
        # Processing the call keyword arguments (line 293)
        kwargs_294074 = {}
        # Getting the type of 'isinstance' (line 293)
        isinstance_294071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 293)
        isinstance_call_result_294075 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), isinstance_294071, *[s_294072, TransferFunction_294073], **kwargs_294074)
        
        # Processing the call keyword arguments (line 293)
        kwargs_294076 = {}
        # Getting the type of 'assert_' (line 293)
        assert__294070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 293)
        assert__call_result_294077 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), assert__294070, *[isinstance_call_result_294075], **kwargs_294076)
        
        
        # Call to assert_(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to isinstance(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 's' (line 294)
        s_294080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 27), 's', False)
        # Getting the type of 'dlti' (line 294)
        dlti_294081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'dlti', False)
        # Processing the call keyword arguments (line 294)
        kwargs_294082 = {}
        # Getting the type of 'isinstance' (line 294)
        isinstance_294079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 294)
        isinstance_call_result_294083 = invoke(stypy.reporting.localization.Localization(__file__, 294, 16), isinstance_294079, *[s_294080, dlti_294081], **kwargs_294082)
        
        # Processing the call keyword arguments (line 294)
        kwargs_294084 = {}
        # Getting the type of 'assert_' (line 294)
        assert__294078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 294)
        assert__call_result_294085 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), assert__294078, *[isinstance_call_result_294083], **kwargs_294084)
        
        
        # Call to assert_(...): (line 295)
        # Processing the call arguments (line 295)
        
        
        # Call to isinstance(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 's' (line 295)
        s_294088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 31), 's', False)
        # Getting the type of 'lti' (line 295)
        lti_294089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 34), 'lti', False)
        # Processing the call keyword arguments (line 295)
        kwargs_294090 = {}
        # Getting the type of 'isinstance' (line 295)
        isinstance_294087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 295)
        isinstance_call_result_294091 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), isinstance_294087, *[s_294088, lti_294089], **kwargs_294090)
        
        # Applying the 'not' unary operator (line 295)
        result_not__294092 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 16), 'not', isinstance_call_result_294091)
        
        # Processing the call keyword arguments (line 295)
        kwargs_294093 = {}
        # Getting the type of 'assert_' (line 295)
        assert__294086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 295)
        assert__call_result_294094 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), assert__294086, *[result_not__294092], **kwargs_294093)
        
        
        # Call to assert_equal(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 's' (line 296)
        s_294096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 's', False)
        # Obtaining the member 'dt' of a type (line 296)
        dt_294097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 21), s_294096, 'dt')
        # Getting the type of 'dt' (line 296)
        dt_294098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'dt', False)
        # Processing the call keyword arguments (line 296)
        kwargs_294099 = {}
        # Getting the type of 'assert_equal' (line 296)
        assert_equal_294095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 296)
        assert_equal_call_result_294100 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert_equal_294095, *[dt_294097, dt_294098], **kwargs_294099)
        
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to dlti(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Call to array(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_294104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        
        # Processing the call keyword arguments (line 299)
        kwargs_294105 = {}
        # Getting the type of 'np' (line 299)
        np_294102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 299)
        array_294103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 17), np_294102, 'array')
        # Calling array(args, kwargs) (line 299)
        array_call_result_294106 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), array_294103, *[list_294104], **kwargs_294105)
        
        
        # Call to array(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'list' (line 299)
        list_294109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 299)
        # Adding element type (line 299)
        int_294110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 40), list_294109, int_294110)
        
        # Processing the call keyword arguments (line 299)
        kwargs_294111 = {}
        # Getting the type of 'np' (line 299)
        np_294107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 31), 'np', False)
        # Obtaining the member 'array' of a type (line 299)
        array_294108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 31), np_294107, 'array')
        # Calling array(args, kwargs) (line 299)
        array_call_result_294112 = invoke(stypy.reporting.localization.Localization(__file__, 299, 31), array_294108, *[list_294109], **kwargs_294111)
        
        int_294113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 47), 'int')
        # Processing the call keyword arguments (line 299)
        # Getting the type of 'dt' (line 299)
        dt_294114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 53), 'dt', False)
        keyword_294115 = dt_294114
        kwargs_294116 = {'dt': keyword_294115}
        # Getting the type of 'dlti' (line 299)
        dlti_294101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'dlti', False)
        # Calling dlti(args, kwargs) (line 299)
        dlti_call_result_294117 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), dlti_294101, *[array_call_result_294106, array_call_result_294112, int_294113], **kwargs_294116)
        
        # Assigning a type to the variable 's' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 's', dlti_call_result_294117)
        
        # Call to assert_(...): (line 300)
        # Processing the call arguments (line 300)
        
        # Call to isinstance(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 's' (line 300)
        s_294120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 27), 's', False)
        # Getting the type of 'ZerosPolesGain' (line 300)
        ZerosPolesGain_294121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'ZerosPolesGain', False)
        # Processing the call keyword arguments (line 300)
        kwargs_294122 = {}
        # Getting the type of 'isinstance' (line 300)
        isinstance_294119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 300)
        isinstance_call_result_294123 = invoke(stypy.reporting.localization.Localization(__file__, 300, 16), isinstance_294119, *[s_294120, ZerosPolesGain_294121], **kwargs_294122)
        
        # Processing the call keyword arguments (line 300)
        kwargs_294124 = {}
        # Getting the type of 'assert_' (line 300)
        assert__294118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 300)
        assert__call_result_294125 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), assert__294118, *[isinstance_call_result_294123], **kwargs_294124)
        
        
        # Call to assert_(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Call to isinstance(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 's' (line 301)
        s_294128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 's', False)
        # Getting the type of 'dlti' (line 301)
        dlti_294129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 30), 'dlti', False)
        # Processing the call keyword arguments (line 301)
        kwargs_294130 = {}
        # Getting the type of 'isinstance' (line 301)
        isinstance_294127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 301)
        isinstance_call_result_294131 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), isinstance_294127, *[s_294128, dlti_294129], **kwargs_294130)
        
        # Processing the call keyword arguments (line 301)
        kwargs_294132 = {}
        # Getting the type of 'assert_' (line 301)
        assert__294126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 301)
        assert__call_result_294133 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), assert__294126, *[isinstance_call_result_294131], **kwargs_294132)
        
        
        # Call to assert_(...): (line 302)
        # Processing the call arguments (line 302)
        
        
        # Call to isinstance(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 's' (line 302)
        s_294136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 31), 's', False)
        # Getting the type of 'lti' (line 302)
        lti_294137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'lti', False)
        # Processing the call keyword arguments (line 302)
        kwargs_294138 = {}
        # Getting the type of 'isinstance' (line 302)
        isinstance_294135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 302)
        isinstance_call_result_294139 = invoke(stypy.reporting.localization.Localization(__file__, 302, 20), isinstance_294135, *[s_294136, lti_294137], **kwargs_294138)
        
        # Applying the 'not' unary operator (line 302)
        result_not__294140 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 16), 'not', isinstance_call_result_294139)
        
        # Processing the call keyword arguments (line 302)
        kwargs_294141 = {}
        # Getting the type of 'assert_' (line 302)
        assert__294134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 302)
        assert__call_result_294142 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), assert__294134, *[result_not__294140], **kwargs_294141)
        
        
        # Call to assert_equal(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 's' (line 303)
        s_294144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 21), 's', False)
        # Obtaining the member 'dt' of a type (line 303)
        dt_294145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 21), s_294144, 'dt')
        # Getting the type of 'dt' (line 303)
        dt_294146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'dt', False)
        # Processing the call keyword arguments (line 303)
        kwargs_294147 = {}
        # Getting the type of 'assert_equal' (line 303)
        assert_equal_294143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 303)
        assert_equal_call_result_294148 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assert_equal_294143, *[dt_294145, dt_294146], **kwargs_294147)
        
        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Assigning a Call to a Name (line 306):
        
        # Call to dlti(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_294150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_294151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 17), list_294150, int_294151)
        
        
        # Obtaining an instance of the builtin type 'list' (line 306)
        list_294152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 306)
        # Adding element type (line 306)
        int_294153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 22), list_294152, int_294153)
        
        int_294154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 28), 'int')
        int_294155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'int')
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'dt' (line 306)
        dt_294156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 37), 'dt', False)
        keyword_294157 = dt_294156
        kwargs_294158 = {'dt': keyword_294157}
        # Getting the type of 'dlti' (line 306)
        dlti_294149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'dlti', False)
        # Calling dlti(args, kwargs) (line 306)
        dlti_call_result_294159 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), dlti_294149, *[list_294150, list_294152, int_294154, int_294155], **kwargs_294158)
        
        # Assigning a type to the variable 's' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 's', dlti_call_result_294159)
        
        # Call to assert_(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to isinstance(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 's' (line 307)
        s_294162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 27), 's', False)
        # Getting the type of 'StateSpace' (line 307)
        StateSpace_294163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), 'StateSpace', False)
        # Processing the call keyword arguments (line 307)
        kwargs_294164 = {}
        # Getting the type of 'isinstance' (line 307)
        isinstance_294161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 307)
        isinstance_call_result_294165 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), isinstance_294161, *[s_294162, StateSpace_294163], **kwargs_294164)
        
        # Processing the call keyword arguments (line 307)
        kwargs_294166 = {}
        # Getting the type of 'assert_' (line 307)
        assert__294160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 307)
        assert__call_result_294167 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), assert__294160, *[isinstance_call_result_294165], **kwargs_294166)
        
        
        # Call to assert_(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Call to isinstance(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 's' (line 308)
        s_294170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 27), 's', False)
        # Getting the type of 'dlti' (line 308)
        dlti_294171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 30), 'dlti', False)
        # Processing the call keyword arguments (line 308)
        kwargs_294172 = {}
        # Getting the type of 'isinstance' (line 308)
        isinstance_294169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 308)
        isinstance_call_result_294173 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), isinstance_294169, *[s_294170, dlti_294171], **kwargs_294172)
        
        # Processing the call keyword arguments (line 308)
        kwargs_294174 = {}
        # Getting the type of 'assert_' (line 308)
        assert__294168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 308)
        assert__call_result_294175 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), assert__294168, *[isinstance_call_result_294173], **kwargs_294174)
        
        
        # Call to assert_(...): (line 309)
        # Processing the call arguments (line 309)
        
        
        # Call to isinstance(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 's' (line 309)
        s_294178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 's', False)
        # Getting the type of 'lti' (line 309)
        lti_294179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 34), 'lti', False)
        # Processing the call keyword arguments (line 309)
        kwargs_294180 = {}
        # Getting the type of 'isinstance' (line 309)
        isinstance_294177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 309)
        isinstance_call_result_294181 = invoke(stypy.reporting.localization.Localization(__file__, 309, 20), isinstance_294177, *[s_294178, lti_294179], **kwargs_294180)
        
        # Applying the 'not' unary operator (line 309)
        result_not__294182 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 16), 'not', isinstance_call_result_294181)
        
        # Processing the call keyword arguments (line 309)
        kwargs_294183 = {}
        # Getting the type of 'assert_' (line 309)
        assert__294176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 309)
        assert__call_result_294184 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), assert__294176, *[result_not__294182], **kwargs_294183)
        
        
        # Call to assert_equal(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 's' (line 310)
        s_294186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 's', False)
        # Obtaining the member 'dt' of a type (line 310)
        dt_294187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 21), s_294186, 'dt')
        # Getting the type of 'dt' (line 310)
        dt_294188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'dt', False)
        # Processing the call keyword arguments (line 310)
        kwargs_294189 = {}
        # Getting the type of 'assert_equal' (line 310)
        assert_equal_294185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 310)
        assert_equal_call_result_294190 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), assert_equal_294185, *[dt_294187, dt_294188], **kwargs_294189)
        
        
        # Call to assert_raises(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'ValueError' (line 313)
        ValueError_294192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 22), 'ValueError', False)
        # Getting the type of 'dlti' (line 313)
        dlti_294193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'dlti', False)
        int_294194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 40), 'int')
        # Processing the call keyword arguments (line 313)
        kwargs_294195 = {}
        # Getting the type of 'assert_raises' (line 313)
        assert_raises_294191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 313)
        assert_raises_call_result_294196 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assert_raises_294191, *[ValueError_294192, dlti_294193, int_294194], **kwargs_294195)
        
        
        # Call to assert_raises(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'ValueError' (line 314)
        ValueError_294198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'ValueError', False)
        # Getting the type of 'dlti' (line 314)
        dlti_294199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'dlti', False)
        int_294200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 40), 'int')
        int_294201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 43), 'int')
        int_294202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 46), 'int')
        int_294203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 49), 'int')
        int_294204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 52), 'int')
        # Processing the call keyword arguments (line 314)
        kwargs_294205 = {}
        # Getting the type of 'assert_raises' (line 314)
        assert_raises_294197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 314)
        assert_raises_call_result_294206 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), assert_raises_294197, *[ValueError_294198, dlti_294199, int_294200, int_294201, int_294202, int_294203, int_294204], **kwargs_294205)
        
        
        # ################# End of 'test_dlti_instantiation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlti_instantiation' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_294207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlti_instantiation'
        return stypy_return_type_294207


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 286, 0, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDlti.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDlti' (line 286)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 0), 'TestDlti', TestDlti)
# Declaration of the 'TestStateSpaceDisc' class

class TestStateSpaceDisc(object, ):

    @norecursion
    def test_initialization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_initialization'
        module_type_store = module_type_store.open_function_context('test_initialization', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_localization', localization)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_function_name', 'TestStateSpaceDisc.test_initialization')
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_param_names_list', [])
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStateSpaceDisc.test_initialization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStateSpaceDisc.test_initialization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_initialization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_initialization(...)' code ##################

        
        # Assigning a Num to a Name (line 320):
        
        # Assigning a Num to a Name (line 320):
        
        # Assigning a Num to a Name (line 320):
        float_294208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 13), 'float')
        # Assigning a type to the variable 'dt' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'dt', float_294208)
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to StateSpace(...): (line 321)
        # Processing the call arguments (line 321)
        int_294210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'int')
        int_294211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 26), 'int')
        int_294212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 29), 'int')
        int_294213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 32), 'int')
        # Processing the call keyword arguments (line 321)
        # Getting the type of 'dt' (line 321)
        dt_294214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 38), 'dt', False)
        keyword_294215 = dt_294214
        kwargs_294216 = {'dt': keyword_294215}
        # Getting the type of 'StateSpace' (line 321)
        StateSpace_294209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 321)
        StateSpace_call_result_294217 = invoke(stypy.reporting.localization.Localization(__file__, 321, 12), StateSpace_294209, *[int_294210, int_294211, int_294212, int_294213], **kwargs_294216)
        
        # Assigning a type to the variable 's' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 's', StateSpace_call_result_294217)
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to StateSpace(...): (line 322)
        # Processing the call arguments (line 322)
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_294219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        int_294220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 23), list_294219, int_294220)
        
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_294221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        int_294222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 28), list_294221, int_294222)
        
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_294223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        int_294224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 33), list_294223, int_294224)
        
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_294225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        int_294226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 38), list_294225, int_294226)
        
        # Processing the call keyword arguments (line 322)
        # Getting the type of 'dt' (line 322)
        dt_294227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 46), 'dt', False)
        keyword_294228 = dt_294227
        kwargs_294229 = {'dt': keyword_294228}
        # Getting the type of 'StateSpace' (line 322)
        StateSpace_294218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 322)
        StateSpace_call_result_294230 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), StateSpace_294218, *[list_294219, list_294221, list_294223, list_294225], **kwargs_294229)
        
        # Assigning a type to the variable 's' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 's', StateSpace_call_result_294230)
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to StateSpace(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Call to array(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_294236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 33), list_294235, int_294236)
        # Adding element type (line 323)
        int_294237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 33), list_294235, int_294237)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 32), list_294234, list_294235)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_294239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 41), list_294238, int_294239)
        # Adding element type (line 323)
        int_294240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 41), list_294238, int_294240)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 32), list_294234, list_294238)
        
        # Processing the call keyword arguments (line 323)
        kwargs_294241 = {}
        # Getting the type of 'np' (line 323)
        np_294232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 323)
        array_294233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 23), np_294232, 'array')
        # Calling array(args, kwargs) (line 323)
        array_call_result_294242 = invoke(stypy.reporting.localization.Localization(__file__, 323, 23), array_294233, *[list_294234], **kwargs_294241)
        
        
        # Call to array(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_294247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 61), list_294246, int_294247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 60), list_294245, list_294246)
        # Adding element type (line 323)
        
        # Obtaining an instance of the builtin type 'list' (line 323)
        list_294248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 323)
        # Adding element type (line 323)
        int_294249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 66), list_294248, int_294249)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 60), list_294245, list_294248)
        
        # Processing the call keyword arguments (line 323)
        kwargs_294250 = {}
        # Getting the type of 'np' (line 323)
        np_294243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 51), 'np', False)
        # Obtaining the member 'array' of a type (line 323)
        array_294244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 51), np_294243, 'array')
        # Calling array(args, kwargs) (line 323)
        array_call_result_294251 = invoke(stypy.reporting.localization.Localization(__file__, 323, 51), array_294244, *[list_294245], **kwargs_294250)
        
        
        # Call to array(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_294254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_294255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_294256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 33), list_294255, int_294256)
        # Adding element type (line 324)
        int_294257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 33), list_294255, int_294257)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 32), list_294254, list_294255)
        
        # Processing the call keyword arguments (line 324)
        kwargs_294258 = {}
        # Getting the type of 'np' (line 324)
        np_294252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'np', False)
        # Obtaining the member 'array' of a type (line 324)
        array_294253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 23), np_294252, 'array')
        # Calling array(args, kwargs) (line 324)
        array_call_result_294259 = invoke(stypy.reporting.localization.Localization(__file__, 324, 23), array_294253, *[list_294254], **kwargs_294258)
        
        
        # Call to array(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_294262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_294263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_294264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 53), list_294263, int_294264)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 52), list_294262, list_294263)
        
        # Processing the call keyword arguments (line 324)
        kwargs_294265 = {}
        # Getting the type of 'np' (line 324)
        np_294260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 324)
        array_294261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 43), np_294260, 'array')
        # Calling array(args, kwargs) (line 324)
        array_call_result_294266 = invoke(stypy.reporting.localization.Localization(__file__, 324, 43), array_294261, *[list_294262], **kwargs_294265)
        
        # Processing the call keyword arguments (line 323)
        # Getting the type of 'dt' (line 324)
        dt_294267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 63), 'dt', False)
        keyword_294268 = dt_294267
        kwargs_294269 = {'dt': keyword_294268}
        # Getting the type of 'StateSpace' (line 323)
        StateSpace_294231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 323)
        StateSpace_call_result_294270 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), StateSpace_294231, *[array_call_result_294242, array_call_result_294251, array_call_result_294259, array_call_result_294266], **kwargs_294269)
        
        # Assigning a type to the variable 's' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 's', StateSpace_call_result_294270)
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to StateSpace(...): (line 325)
        # Processing the call arguments (line 325)
        int_294272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 23), 'int')
        int_294273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 26), 'int')
        int_294274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 29), 'int')
        int_294275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 32), 'int')
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'True' (line 325)
        True_294276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'True', False)
        keyword_294277 = True_294276
        kwargs_294278 = {'dt': keyword_294277}
        # Getting the type of 'StateSpace' (line 325)
        StateSpace_294271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 325)
        StateSpace_call_result_294279 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), StateSpace_294271, *[int_294272, int_294273, int_294274, int_294275], **kwargs_294278)
        
        # Assigning a type to the variable 's' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 's', StateSpace_call_result_294279)
        
        # ################# End of 'test_initialization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_initialization' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_294280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_initialization'
        return stypy_return_type_294280


    @norecursion
    def test_conversion(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_conversion'
        module_type_store = module_type_store.open_function_context('test_conversion', 327, 4, False)
        # Assigning a type to the variable 'self' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_localization', localization)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_function_name', 'TestStateSpaceDisc.test_conversion')
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_param_names_list', [])
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStateSpaceDisc.test_conversion.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStateSpaceDisc.test_conversion', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_conversion', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_conversion(...)' code ##################

        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to StateSpace(...): (line 329)
        # Processing the call arguments (line 329)
        int_294282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 23), 'int')
        int_294283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 26), 'int')
        int_294284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 29), 'int')
        int_294285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 32), 'int')
        # Processing the call keyword arguments (line 329)
        float_294286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 38), 'float')
        keyword_294287 = float_294286
        kwargs_294288 = {'dt': keyword_294287}
        # Getting the type of 'StateSpace' (line 329)
        StateSpace_294281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 329)
        StateSpace_call_result_294289 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), StateSpace_294281, *[int_294282, int_294283, int_294284, int_294285], **kwargs_294288)
        
        # Assigning a type to the variable 's' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 's', StateSpace_call_result_294289)
        
        # Call to assert_(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Call to isinstance(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Call to to_ss(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_294294 = {}
        # Getting the type of 's' (line 330)
        s_294292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 's', False)
        # Obtaining the member 'to_ss' of a type (line 330)
        to_ss_294293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 27), s_294292, 'to_ss')
        # Calling to_ss(args, kwargs) (line 330)
        to_ss_call_result_294295 = invoke(stypy.reporting.localization.Localization(__file__, 330, 27), to_ss_294293, *[], **kwargs_294294)
        
        # Getting the type of 'StateSpace' (line 330)
        StateSpace_294296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 38), 'StateSpace', False)
        # Processing the call keyword arguments (line 330)
        kwargs_294297 = {}
        # Getting the type of 'isinstance' (line 330)
        isinstance_294291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 330)
        isinstance_call_result_294298 = invoke(stypy.reporting.localization.Localization(__file__, 330, 16), isinstance_294291, *[to_ss_call_result_294295, StateSpace_294296], **kwargs_294297)
        
        # Processing the call keyword arguments (line 330)
        kwargs_294299 = {}
        # Getting the type of 'assert_' (line 330)
        assert__294290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 330)
        assert__call_result_294300 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), assert__294290, *[isinstance_call_result_294298], **kwargs_294299)
        
        
        # Call to assert_(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Call to isinstance(...): (line 331)
        # Processing the call arguments (line 331)
        
        # Call to to_tf(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_294305 = {}
        # Getting the type of 's' (line 331)
        s_294303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 's', False)
        # Obtaining the member 'to_tf' of a type (line 331)
        to_tf_294304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 27), s_294303, 'to_tf')
        # Calling to_tf(args, kwargs) (line 331)
        to_tf_call_result_294306 = invoke(stypy.reporting.localization.Localization(__file__, 331, 27), to_tf_294304, *[], **kwargs_294305)
        
        # Getting the type of 'TransferFunction' (line 331)
        TransferFunction_294307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 38), 'TransferFunction', False)
        # Processing the call keyword arguments (line 331)
        kwargs_294308 = {}
        # Getting the type of 'isinstance' (line 331)
        isinstance_294302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 331)
        isinstance_call_result_294309 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), isinstance_294302, *[to_tf_call_result_294306, TransferFunction_294307], **kwargs_294308)
        
        # Processing the call keyword arguments (line 331)
        kwargs_294310 = {}
        # Getting the type of 'assert_' (line 331)
        assert__294301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 331)
        assert__call_result_294311 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), assert__294301, *[isinstance_call_result_294309], **kwargs_294310)
        
        
        # Call to assert_(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to isinstance(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to to_zpk(...): (line 332)
        # Processing the call keyword arguments (line 332)
        kwargs_294316 = {}
        # Getting the type of 's' (line 332)
        s_294314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 27), 's', False)
        # Obtaining the member 'to_zpk' of a type (line 332)
        to_zpk_294315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 27), s_294314, 'to_zpk')
        # Calling to_zpk(args, kwargs) (line 332)
        to_zpk_call_result_294317 = invoke(stypy.reporting.localization.Localization(__file__, 332, 27), to_zpk_294315, *[], **kwargs_294316)
        
        # Getting the type of 'ZerosPolesGain' (line 332)
        ZerosPolesGain_294318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 39), 'ZerosPolesGain', False)
        # Processing the call keyword arguments (line 332)
        kwargs_294319 = {}
        # Getting the type of 'isinstance' (line 332)
        isinstance_294313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 332)
        isinstance_call_result_294320 = invoke(stypy.reporting.localization.Localization(__file__, 332, 16), isinstance_294313, *[to_zpk_call_result_294317, ZerosPolesGain_294318], **kwargs_294319)
        
        # Processing the call keyword arguments (line 332)
        kwargs_294321 = {}
        # Getting the type of 'assert_' (line 332)
        assert__294312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 332)
        assert__call_result_294322 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), assert__294312, *[isinstance_call_result_294320], **kwargs_294321)
        
        
        # Call to assert_(...): (line 335)
        # Processing the call arguments (line 335)
        
        
        # Call to StateSpace(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 's' (line 335)
        s_294325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 27), 's', False)
        # Processing the call keyword arguments (line 335)
        kwargs_294326 = {}
        # Getting the type of 'StateSpace' (line 335)
        StateSpace_294324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 16), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 335)
        StateSpace_call_result_294327 = invoke(stypy.reporting.localization.Localization(__file__, 335, 16), StateSpace_294324, *[s_294325], **kwargs_294326)
        
        # Getting the type of 's' (line 335)
        s_294328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 37), 's', False)
        # Applying the binary operator 'isnot' (line 335)
        result_is_not_294329 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 16), 'isnot', StateSpace_call_result_294327, s_294328)
        
        # Processing the call keyword arguments (line 335)
        kwargs_294330 = {}
        # Getting the type of 'assert_' (line 335)
        assert__294323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 335)
        assert__call_result_294331 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), assert__294323, *[result_is_not_294329], **kwargs_294330)
        
        
        # Call to assert_(...): (line 336)
        # Processing the call arguments (line 336)
        
        
        # Call to to_ss(...): (line 336)
        # Processing the call keyword arguments (line 336)
        kwargs_294335 = {}
        # Getting the type of 's' (line 336)
        s_294333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 's', False)
        # Obtaining the member 'to_ss' of a type (line 336)
        to_ss_294334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), s_294333, 'to_ss')
        # Calling to_ss(args, kwargs) (line 336)
        to_ss_call_result_294336 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), to_ss_294334, *[], **kwargs_294335)
        
        # Getting the type of 's' (line 336)
        s_294337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 's', False)
        # Applying the binary operator 'isnot' (line 336)
        result_is_not_294338 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 16), 'isnot', to_ss_call_result_294336, s_294337)
        
        # Processing the call keyword arguments (line 336)
        kwargs_294339 = {}
        # Getting the type of 'assert_' (line 336)
        assert__294332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 336)
        assert__call_result_294340 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), assert__294332, *[result_is_not_294338], **kwargs_294339)
        
        
        # ################# End of 'test_conversion(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_conversion' in the type store
        # Getting the type of 'stypy_return_type' (line 327)
        stypy_return_type_294341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294341)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_conversion'
        return stypy_return_type_294341


    @norecursion
    def test_properties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_properties'
        module_type_store = module_type_store.open_function_context('test_properties', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_localization', localization)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_function_name', 'TestStateSpaceDisc.test_properties')
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_param_names_list', [])
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStateSpaceDisc.test_properties.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStateSpaceDisc.test_properties', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_properties', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_properties(...)' code ##################

        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to StateSpace(...): (line 343)
        # Processing the call arguments (line 343)
        int_294343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'int')
        int_294344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 26), 'int')
        int_294345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 29), 'int')
        int_294346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 32), 'int')
        # Processing the call keyword arguments (line 343)
        float_294347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 38), 'float')
        keyword_294348 = float_294347
        kwargs_294349 = {'dt': keyword_294348}
        # Getting the type of 'StateSpace' (line 343)
        StateSpace_294342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'StateSpace', False)
        # Calling StateSpace(args, kwargs) (line 343)
        StateSpace_call_result_294350 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), StateSpace_294342, *[int_294343, int_294344, int_294345, int_294346], **kwargs_294349)
        
        # Assigning a type to the variable 's' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 's', StateSpace_call_result_294350)
        
        # Call to assert_equal(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 's' (line 344)
        s_294352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 's', False)
        # Obtaining the member 'poles' of a type (line 344)
        poles_294353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 21), s_294352, 'poles')
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_294354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_294355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 30), list_294354, int_294355)
        
        # Processing the call keyword arguments (line 344)
        kwargs_294356 = {}
        # Getting the type of 'assert_equal' (line 344)
        assert_equal_294351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 344)
        assert_equal_call_result_294357 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), assert_equal_294351, *[poles_294353, list_294354], **kwargs_294356)
        
        
        # Call to assert_equal(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 's' (line 345)
        s_294359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 's', False)
        # Obtaining the member 'zeros' of a type (line 345)
        zeros_294360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 21), s_294359, 'zeros')
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_294361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        int_294362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 30), list_294361, int_294362)
        
        # Processing the call keyword arguments (line 345)
        kwargs_294363 = {}
        # Getting the type of 'assert_equal' (line 345)
        assert_equal_294358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 345)
        assert_equal_call_result_294364 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_equal_294358, *[zeros_294360, list_294361], **kwargs_294363)
        
        
        # ################# End of 'test_properties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_properties' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_294365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_properties'
        return stypy_return_type_294365


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStateSpaceDisc.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestStateSpaceDisc' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'TestStateSpaceDisc', TestStateSpaceDisc)
# Declaration of the 'TestTransferFunction' class

class TestTransferFunction(object, ):

    @norecursion
    def test_initialization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_initialization'
        module_type_store = module_type_store.open_function_context('test_initialization', 349, 4, False)
        # Assigning a type to the variable 'self' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_function_name', 'TestTransferFunction.test_initialization')
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunction.test_initialization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunction.test_initialization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_initialization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_initialization(...)' code ##################

        
        # Assigning a Num to a Name (line 351):
        
        # Assigning a Num to a Name (line 351):
        
        # Assigning a Num to a Name (line 351):
        float_294366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 13), 'float')
        # Assigning a type to the variable 'dt' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'dt', float_294366)
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to TransferFunction(...): (line 352)
        # Processing the call arguments (line 352)
        int_294368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 29), 'int')
        int_294369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 32), 'int')
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'dt' (line 352)
        dt_294370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'dt', False)
        keyword_294371 = dt_294370
        kwargs_294372 = {'dt': keyword_294371}
        # Getting the type of 'TransferFunction' (line 352)
        TransferFunction_294367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 352)
        TransferFunction_call_result_294373 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), TransferFunction_294367, *[int_294368, int_294369], **kwargs_294372)
        
        # Assigning a type to the variable 's' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 's', TransferFunction_call_result_294373)
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to TransferFunction(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_294375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        # Adding element type (line 353)
        int_294376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 29), list_294375, int_294376)
        
        
        # Obtaining an instance of the builtin type 'list' (line 353)
        list_294377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 353)
        # Adding element type (line 353)
        int_294378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 34), list_294377, int_294378)
        
        # Processing the call keyword arguments (line 353)
        # Getting the type of 'dt' (line 353)
        dt_294379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 42), 'dt', False)
        keyword_294380 = dt_294379
        kwargs_294381 = {'dt': keyword_294380}
        # Getting the type of 'TransferFunction' (line 353)
        TransferFunction_294374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 353)
        TransferFunction_call_result_294382 = invoke(stypy.reporting.localization.Localization(__file__, 353, 12), TransferFunction_294374, *[list_294375, list_294377], **kwargs_294381)
        
        # Assigning a type to the variable 's' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 's', TransferFunction_call_result_294382)
        
        # Assigning a Call to a Name (line 354):
        
        # Assigning a Call to a Name (line 354):
        
        # Assigning a Call to a Name (line 354):
        
        # Call to TransferFunction(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to array(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Obtaining an instance of the builtin type 'list' (line 354)
        list_294386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 354)
        # Adding element type (line 354)
        int_294387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 38), list_294386, int_294387)
        
        # Processing the call keyword arguments (line 354)
        kwargs_294388 = {}
        # Getting the type of 'np' (line 354)
        np_294384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'np', False)
        # Obtaining the member 'array' of a type (line 354)
        array_294385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 29), np_294384, 'array')
        # Calling array(args, kwargs) (line 354)
        array_call_result_294389 = invoke(stypy.reporting.localization.Localization(__file__, 354, 29), array_294385, *[list_294386], **kwargs_294388)
        
        
        # Call to array(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Obtaining an instance of the builtin type 'list' (line 354)
        list_294392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 354)
        # Adding element type (line 354)
        int_294393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 53), list_294392, int_294393)
        
        # Processing the call keyword arguments (line 354)
        kwargs_294394 = {}
        # Getting the type of 'np' (line 354)
        np_294390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 354)
        array_294391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 44), np_294390, 'array')
        # Calling array(args, kwargs) (line 354)
        array_call_result_294395 = invoke(stypy.reporting.localization.Localization(__file__, 354, 44), array_294391, *[list_294392], **kwargs_294394)
        
        # Processing the call keyword arguments (line 354)
        # Getting the type of 'dt' (line 354)
        dt_294396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 62), 'dt', False)
        keyword_294397 = dt_294396
        kwargs_294398 = {'dt': keyword_294397}
        # Getting the type of 'TransferFunction' (line 354)
        TransferFunction_294383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 354)
        TransferFunction_call_result_294399 = invoke(stypy.reporting.localization.Localization(__file__, 354, 12), TransferFunction_294383, *[array_call_result_294389, array_call_result_294395], **kwargs_294398)
        
        # Assigning a type to the variable 's' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 's', TransferFunction_call_result_294399)
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to TransferFunction(...): (line 355)
        # Processing the call arguments (line 355)
        int_294401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 29), 'int')
        int_294402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 32), 'int')
        # Processing the call keyword arguments (line 355)
        # Getting the type of 'True' (line 355)
        True_294403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 38), 'True', False)
        keyword_294404 = True_294403
        kwargs_294405 = {'dt': keyword_294404}
        # Getting the type of 'TransferFunction' (line 355)
        TransferFunction_294400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 355)
        TransferFunction_call_result_294406 = invoke(stypy.reporting.localization.Localization(__file__, 355, 12), TransferFunction_294400, *[int_294401, int_294402], **kwargs_294405)
        
        # Assigning a type to the variable 's' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 's', TransferFunction_call_result_294406)
        
        # ################# End of 'test_initialization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_initialization' in the type store
        # Getting the type of 'stypy_return_type' (line 349)
        stypy_return_type_294407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294407)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_initialization'
        return stypy_return_type_294407


    @norecursion
    def test_conversion(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_conversion'
        module_type_store = module_type_store.open_function_context('test_conversion', 357, 4, False)
        # Assigning a type to the variable 'self' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_function_name', 'TestTransferFunction.test_conversion')
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunction.test_conversion.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunction.test_conversion', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_conversion', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_conversion(...)' code ##################

        
        # Assigning a Call to a Name (line 359):
        
        # Assigning a Call to a Name (line 359):
        
        # Assigning a Call to a Name (line 359):
        
        # Call to TransferFunction(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Obtaining an instance of the builtin type 'list' (line 359)
        list_294409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 359)
        # Adding element type (line 359)
        int_294410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 29), list_294409, int_294410)
        # Adding element type (line 359)
        int_294411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 29), list_294409, int_294411)
        
        
        # Obtaining an instance of the builtin type 'list' (line 359)
        list_294412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 359)
        # Adding element type (line 359)
        int_294413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 37), list_294412, int_294413)
        # Adding element type (line 359)
        int_294414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 37), list_294412, int_294414)
        
        # Processing the call keyword arguments (line 359)
        float_294415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 49), 'float')
        keyword_294416 = float_294415
        kwargs_294417 = {'dt': keyword_294416}
        # Getting the type of 'TransferFunction' (line 359)
        TransferFunction_294408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 359)
        TransferFunction_call_result_294418 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), TransferFunction_294408, *[list_294409, list_294412], **kwargs_294417)
        
        # Assigning a type to the variable 's' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 's', TransferFunction_call_result_294418)
        
        # Call to assert_(...): (line 360)
        # Processing the call arguments (line 360)
        
        # Call to isinstance(...): (line 360)
        # Processing the call arguments (line 360)
        
        # Call to to_ss(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_294423 = {}
        # Getting the type of 's' (line 360)
        s_294421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 's', False)
        # Obtaining the member 'to_ss' of a type (line 360)
        to_ss_294422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 27), s_294421, 'to_ss')
        # Calling to_ss(args, kwargs) (line 360)
        to_ss_call_result_294424 = invoke(stypy.reporting.localization.Localization(__file__, 360, 27), to_ss_294422, *[], **kwargs_294423)
        
        # Getting the type of 'StateSpace' (line 360)
        StateSpace_294425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 38), 'StateSpace', False)
        # Processing the call keyword arguments (line 360)
        kwargs_294426 = {}
        # Getting the type of 'isinstance' (line 360)
        isinstance_294420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 360)
        isinstance_call_result_294427 = invoke(stypy.reporting.localization.Localization(__file__, 360, 16), isinstance_294420, *[to_ss_call_result_294424, StateSpace_294425], **kwargs_294426)
        
        # Processing the call keyword arguments (line 360)
        kwargs_294428 = {}
        # Getting the type of 'assert_' (line 360)
        assert__294419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 360)
        assert__call_result_294429 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assert__294419, *[isinstance_call_result_294427], **kwargs_294428)
        
        
        # Call to assert_(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Call to isinstance(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Call to to_tf(...): (line 361)
        # Processing the call keyword arguments (line 361)
        kwargs_294434 = {}
        # Getting the type of 's' (line 361)
        s_294432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 's', False)
        # Obtaining the member 'to_tf' of a type (line 361)
        to_tf_294433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 27), s_294432, 'to_tf')
        # Calling to_tf(args, kwargs) (line 361)
        to_tf_call_result_294435 = invoke(stypy.reporting.localization.Localization(__file__, 361, 27), to_tf_294433, *[], **kwargs_294434)
        
        # Getting the type of 'TransferFunction' (line 361)
        TransferFunction_294436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 38), 'TransferFunction', False)
        # Processing the call keyword arguments (line 361)
        kwargs_294437 = {}
        # Getting the type of 'isinstance' (line 361)
        isinstance_294431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 361)
        isinstance_call_result_294438 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), isinstance_294431, *[to_tf_call_result_294435, TransferFunction_294436], **kwargs_294437)
        
        # Processing the call keyword arguments (line 361)
        kwargs_294439 = {}
        # Getting the type of 'assert_' (line 361)
        assert__294430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 361)
        assert__call_result_294440 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), assert__294430, *[isinstance_call_result_294438], **kwargs_294439)
        
        
        # Call to assert_(...): (line 362)
        # Processing the call arguments (line 362)
        
        # Call to isinstance(...): (line 362)
        # Processing the call arguments (line 362)
        
        # Call to to_zpk(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_294445 = {}
        # Getting the type of 's' (line 362)
        s_294443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 's', False)
        # Obtaining the member 'to_zpk' of a type (line 362)
        to_zpk_294444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 27), s_294443, 'to_zpk')
        # Calling to_zpk(args, kwargs) (line 362)
        to_zpk_call_result_294446 = invoke(stypy.reporting.localization.Localization(__file__, 362, 27), to_zpk_294444, *[], **kwargs_294445)
        
        # Getting the type of 'ZerosPolesGain' (line 362)
        ZerosPolesGain_294447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 39), 'ZerosPolesGain', False)
        # Processing the call keyword arguments (line 362)
        kwargs_294448 = {}
        # Getting the type of 'isinstance' (line 362)
        isinstance_294442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 362)
        isinstance_call_result_294449 = invoke(stypy.reporting.localization.Localization(__file__, 362, 16), isinstance_294442, *[to_zpk_call_result_294446, ZerosPolesGain_294447], **kwargs_294448)
        
        # Processing the call keyword arguments (line 362)
        kwargs_294450 = {}
        # Getting the type of 'assert_' (line 362)
        assert__294441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 362)
        assert__call_result_294451 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), assert__294441, *[isinstance_call_result_294449], **kwargs_294450)
        
        
        # Call to assert_(...): (line 365)
        # Processing the call arguments (line 365)
        
        
        # Call to TransferFunction(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 's' (line 365)
        s_294454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 33), 's', False)
        # Processing the call keyword arguments (line 365)
        kwargs_294455 = {}
        # Getting the type of 'TransferFunction' (line 365)
        TransferFunction_294453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 365)
        TransferFunction_call_result_294456 = invoke(stypy.reporting.localization.Localization(__file__, 365, 16), TransferFunction_294453, *[s_294454], **kwargs_294455)
        
        # Getting the type of 's' (line 365)
        s_294457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 43), 's', False)
        # Applying the binary operator 'isnot' (line 365)
        result_is_not_294458 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 16), 'isnot', TransferFunction_call_result_294456, s_294457)
        
        # Processing the call keyword arguments (line 365)
        kwargs_294459 = {}
        # Getting the type of 'assert_' (line 365)
        assert__294452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 365)
        assert__call_result_294460 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), assert__294452, *[result_is_not_294458], **kwargs_294459)
        
        
        # Call to assert_(...): (line 366)
        # Processing the call arguments (line 366)
        
        
        # Call to to_tf(...): (line 366)
        # Processing the call keyword arguments (line 366)
        kwargs_294464 = {}
        # Getting the type of 's' (line 366)
        s_294462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 's', False)
        # Obtaining the member 'to_tf' of a type (line 366)
        to_tf_294463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), s_294462, 'to_tf')
        # Calling to_tf(args, kwargs) (line 366)
        to_tf_call_result_294465 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), to_tf_294463, *[], **kwargs_294464)
        
        # Getting the type of 's' (line 366)
        s_294466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 's', False)
        # Applying the binary operator 'isnot' (line 366)
        result_is_not_294467 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 16), 'isnot', to_tf_call_result_294465, s_294466)
        
        # Processing the call keyword arguments (line 366)
        kwargs_294468 = {}
        # Getting the type of 'assert_' (line 366)
        assert__294461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 366)
        assert__call_result_294469 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), assert__294461, *[result_is_not_294467], **kwargs_294468)
        
        
        # ################# End of 'test_conversion(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_conversion' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_294470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294470)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_conversion'
        return stypy_return_type_294470


    @norecursion
    def test_properties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_properties'
        module_type_store = module_type_store.open_function_context('test_properties', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_function_name', 'TestTransferFunction.test_properties')
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunction.test_properties.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunction.test_properties', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_properties', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_properties(...)' code ##################

        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to TransferFunction(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_294472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        int_294473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 29), list_294472, int_294473)
        # Adding element type (line 373)
        int_294474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 29), list_294472, int_294474)
        
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_294475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        int_294476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 37), list_294475, int_294476)
        # Adding element type (line 373)
        int_294477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 37), list_294475, int_294477)
        
        # Processing the call keyword arguments (line 373)
        float_294478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 49), 'float')
        keyword_294479 = float_294478
        kwargs_294480 = {'dt': keyword_294479}
        # Getting the type of 'TransferFunction' (line 373)
        TransferFunction_294471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 373)
        TransferFunction_call_result_294481 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), TransferFunction_294471, *[list_294472, list_294475], **kwargs_294480)
        
        # Assigning a type to the variable 's' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 's', TransferFunction_call_result_294481)
        
        # Call to assert_equal(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 's' (line 374)
        s_294483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 's', False)
        # Obtaining the member 'poles' of a type (line 374)
        poles_294484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 21), s_294483, 'poles')
        
        # Obtaining an instance of the builtin type 'list' (line 374)
        list_294485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 374)
        # Adding element type (line 374)
        int_294486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 30), list_294485, int_294486)
        
        # Processing the call keyword arguments (line 374)
        kwargs_294487 = {}
        # Getting the type of 'assert_equal' (line 374)
        assert_equal_294482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 374)
        assert_equal_call_result_294488 = invoke(stypy.reporting.localization.Localization(__file__, 374, 8), assert_equal_294482, *[poles_294484, list_294485], **kwargs_294487)
        
        
        # Call to assert_equal(...): (line 375)
        # Processing the call arguments (line 375)
        # Getting the type of 's' (line 375)
        s_294490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 21), 's', False)
        # Obtaining the member 'zeros' of a type (line 375)
        zeros_294491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 21), s_294490, 'zeros')
        
        # Obtaining an instance of the builtin type 'list' (line 375)
        list_294492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 375)
        # Adding element type (line 375)
        int_294493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 30), list_294492, int_294493)
        
        # Processing the call keyword arguments (line 375)
        kwargs_294494 = {}
        # Getting the type of 'assert_equal' (line 375)
        assert_equal_294489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 375)
        assert_equal_call_result_294495 = invoke(stypy.reporting.localization.Localization(__file__, 375, 8), assert_equal_294489, *[zeros_294491, list_294492], **kwargs_294494)
        
        
        # ################# End of 'test_properties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_properties' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_294496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_properties'
        return stypy_return_type_294496


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunction.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTransferFunction' (line 348)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 0), 'TestTransferFunction', TestTransferFunction)
# Declaration of the 'TestZerosPolesGain' class

class TestZerosPolesGain(object, ):

    @norecursion
    def test_initialization(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_initialization'
        module_type_store = module_type_store.open_function_context('test_initialization', 379, 4, False)
        # Assigning a type to the variable 'self' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_localization', localization)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_function_name', 'TestZerosPolesGain.test_initialization')
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_param_names_list', [])
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZerosPolesGain.test_initialization.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZerosPolesGain.test_initialization', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_initialization', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_initialization(...)' code ##################

        
        # Assigning a Num to a Name (line 381):
        
        # Assigning a Num to a Name (line 381):
        
        # Assigning a Num to a Name (line 381):
        float_294497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 13), 'float')
        # Assigning a type to the variable 'dt' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'dt', float_294497)
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to ZerosPolesGain(...): (line 382)
        # Processing the call arguments (line 382)
        int_294499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 27), 'int')
        int_294500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 30), 'int')
        int_294501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 33), 'int')
        # Processing the call keyword arguments (line 382)
        # Getting the type of 'dt' (line 382)
        dt_294502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 39), 'dt', False)
        keyword_294503 = dt_294502
        kwargs_294504 = {'dt': keyword_294503}
        # Getting the type of 'ZerosPolesGain' (line 382)
        ZerosPolesGain_294498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 382)
        ZerosPolesGain_call_result_294505 = invoke(stypy.reporting.localization.Localization(__file__, 382, 12), ZerosPolesGain_294498, *[int_294499, int_294500, int_294501], **kwargs_294504)
        
        # Assigning a type to the variable 's' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 's', ZerosPolesGain_call_result_294505)
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to ZerosPolesGain(...): (line 383)
        # Processing the call arguments (line 383)
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_294507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        # Adding element type (line 383)
        int_294508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 27), list_294507, int_294508)
        
        
        # Obtaining an instance of the builtin type 'list' (line 383)
        list_294509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 383)
        # Adding element type (line 383)
        int_294510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 32), list_294509, int_294510)
        
        int_294511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 37), 'int')
        # Processing the call keyword arguments (line 383)
        # Getting the type of 'dt' (line 383)
        dt_294512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 43), 'dt', False)
        keyword_294513 = dt_294512
        kwargs_294514 = {'dt': keyword_294513}
        # Getting the type of 'ZerosPolesGain' (line 383)
        ZerosPolesGain_294506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 383)
        ZerosPolesGain_call_result_294515 = invoke(stypy.reporting.localization.Localization(__file__, 383, 12), ZerosPolesGain_294506, *[list_294507, list_294509, int_294511], **kwargs_294514)
        
        # Assigning a type to the variable 's' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 's', ZerosPolesGain_call_result_294515)
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to ZerosPolesGain(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Call to array(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_294519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_294520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 36), list_294519, int_294520)
        
        # Processing the call keyword arguments (line 384)
        kwargs_294521 = {}
        # Getting the type of 'np' (line 384)
        np_294517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'np', False)
        # Obtaining the member 'array' of a type (line 384)
        array_294518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 27), np_294517, 'array')
        # Calling array(args, kwargs) (line 384)
        array_call_result_294522 = invoke(stypy.reporting.localization.Localization(__file__, 384, 27), array_294518, *[list_294519], **kwargs_294521)
        
        
        # Call to array(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Obtaining an instance of the builtin type 'list' (line 384)
        list_294525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 384)
        # Adding element type (line 384)
        int_294526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 51), list_294525, int_294526)
        
        # Processing the call keyword arguments (line 384)
        kwargs_294527 = {}
        # Getting the type of 'np' (line 384)
        np_294523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 42), 'np', False)
        # Obtaining the member 'array' of a type (line 384)
        array_294524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 42), np_294523, 'array')
        # Calling array(args, kwargs) (line 384)
        array_call_result_294528 = invoke(stypy.reporting.localization.Localization(__file__, 384, 42), array_294524, *[list_294525], **kwargs_294527)
        
        int_294529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 57), 'int')
        # Processing the call keyword arguments (line 384)
        # Getting the type of 'dt' (line 384)
        dt_294530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 63), 'dt', False)
        keyword_294531 = dt_294530
        kwargs_294532 = {'dt': keyword_294531}
        # Getting the type of 'ZerosPolesGain' (line 384)
        ZerosPolesGain_294516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 384)
        ZerosPolesGain_call_result_294533 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), ZerosPolesGain_294516, *[array_call_result_294522, array_call_result_294528, int_294529], **kwargs_294532)
        
        # Assigning a type to the variable 's' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 's', ZerosPolesGain_call_result_294533)
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to ZerosPolesGain(...): (line 385)
        # Processing the call arguments (line 385)
        int_294535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 27), 'int')
        int_294536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'int')
        int_294537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 33), 'int')
        # Processing the call keyword arguments (line 385)
        # Getting the type of 'True' (line 385)
        True_294538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 39), 'True', False)
        keyword_294539 = True_294538
        kwargs_294540 = {'dt': keyword_294539}
        # Getting the type of 'ZerosPolesGain' (line 385)
        ZerosPolesGain_294534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 385)
        ZerosPolesGain_call_result_294541 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), ZerosPolesGain_294534, *[int_294535, int_294536, int_294537], **kwargs_294540)
        
        # Assigning a type to the variable 's' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 's', ZerosPolesGain_call_result_294541)
        
        # ################# End of 'test_initialization(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_initialization' in the type store
        # Getting the type of 'stypy_return_type' (line 379)
        stypy_return_type_294542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_initialization'
        return stypy_return_type_294542


    @norecursion
    def test_conversion(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_conversion'
        module_type_store = module_type_store.open_function_context('test_conversion', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_localization', localization)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_function_name', 'TestZerosPolesGain.test_conversion')
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_param_names_list', [])
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestZerosPolesGain.test_conversion.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZerosPolesGain.test_conversion', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_conversion', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_conversion(...)' code ##################

        
        # Assigning a Call to a Name (line 389):
        
        # Assigning a Call to a Name (line 389):
        
        # Assigning a Call to a Name (line 389):
        
        # Call to ZerosPolesGain(...): (line 389)
        # Processing the call arguments (line 389)
        int_294544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 27), 'int')
        int_294545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 30), 'int')
        int_294546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 33), 'int')
        # Processing the call keyword arguments (line 389)
        float_294547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 39), 'float')
        keyword_294548 = float_294547
        kwargs_294549 = {'dt': keyword_294548}
        # Getting the type of 'ZerosPolesGain' (line 389)
        ZerosPolesGain_294543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 389)
        ZerosPolesGain_call_result_294550 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), ZerosPolesGain_294543, *[int_294544, int_294545, int_294546], **kwargs_294549)
        
        # Assigning a type to the variable 's' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 's', ZerosPolesGain_call_result_294550)
        
        # Call to assert_(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Call to isinstance(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Call to to_ss(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_294555 = {}
        # Getting the type of 's' (line 390)
        s_294553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 27), 's', False)
        # Obtaining the member 'to_ss' of a type (line 390)
        to_ss_294554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 27), s_294553, 'to_ss')
        # Calling to_ss(args, kwargs) (line 390)
        to_ss_call_result_294556 = invoke(stypy.reporting.localization.Localization(__file__, 390, 27), to_ss_294554, *[], **kwargs_294555)
        
        # Getting the type of 'StateSpace' (line 390)
        StateSpace_294557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 38), 'StateSpace', False)
        # Processing the call keyword arguments (line 390)
        kwargs_294558 = {}
        # Getting the type of 'isinstance' (line 390)
        isinstance_294552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 390)
        isinstance_call_result_294559 = invoke(stypy.reporting.localization.Localization(__file__, 390, 16), isinstance_294552, *[to_ss_call_result_294556, StateSpace_294557], **kwargs_294558)
        
        # Processing the call keyword arguments (line 390)
        kwargs_294560 = {}
        # Getting the type of 'assert_' (line 390)
        assert__294551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 390)
        assert__call_result_294561 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), assert__294551, *[isinstance_call_result_294559], **kwargs_294560)
        
        
        # Call to assert_(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Call to isinstance(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Call to to_tf(...): (line 391)
        # Processing the call keyword arguments (line 391)
        kwargs_294566 = {}
        # Getting the type of 's' (line 391)
        s_294564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 27), 's', False)
        # Obtaining the member 'to_tf' of a type (line 391)
        to_tf_294565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 27), s_294564, 'to_tf')
        # Calling to_tf(args, kwargs) (line 391)
        to_tf_call_result_294567 = invoke(stypy.reporting.localization.Localization(__file__, 391, 27), to_tf_294565, *[], **kwargs_294566)
        
        # Getting the type of 'TransferFunction' (line 391)
        TransferFunction_294568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 38), 'TransferFunction', False)
        # Processing the call keyword arguments (line 391)
        kwargs_294569 = {}
        # Getting the type of 'isinstance' (line 391)
        isinstance_294563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 391)
        isinstance_call_result_294570 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), isinstance_294563, *[to_tf_call_result_294567, TransferFunction_294568], **kwargs_294569)
        
        # Processing the call keyword arguments (line 391)
        kwargs_294571 = {}
        # Getting the type of 'assert_' (line 391)
        assert__294562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 391)
        assert__call_result_294572 = invoke(stypy.reporting.localization.Localization(__file__, 391, 8), assert__294562, *[isinstance_call_result_294570], **kwargs_294571)
        
        
        # Call to assert_(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Call to isinstance(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Call to to_zpk(...): (line 392)
        # Processing the call keyword arguments (line 392)
        kwargs_294577 = {}
        # Getting the type of 's' (line 392)
        s_294575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 's', False)
        # Obtaining the member 'to_zpk' of a type (line 392)
        to_zpk_294576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 27), s_294575, 'to_zpk')
        # Calling to_zpk(args, kwargs) (line 392)
        to_zpk_call_result_294578 = invoke(stypy.reporting.localization.Localization(__file__, 392, 27), to_zpk_294576, *[], **kwargs_294577)
        
        # Getting the type of 'ZerosPolesGain' (line 392)
        ZerosPolesGain_294579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 39), 'ZerosPolesGain', False)
        # Processing the call keyword arguments (line 392)
        kwargs_294580 = {}
        # Getting the type of 'isinstance' (line 392)
        isinstance_294574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 392)
        isinstance_call_result_294581 = invoke(stypy.reporting.localization.Localization(__file__, 392, 16), isinstance_294574, *[to_zpk_call_result_294578, ZerosPolesGain_294579], **kwargs_294580)
        
        # Processing the call keyword arguments (line 392)
        kwargs_294582 = {}
        # Getting the type of 'assert_' (line 392)
        assert__294573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 392)
        assert__call_result_294583 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), assert__294573, *[isinstance_call_result_294581], **kwargs_294582)
        
        
        # Call to assert_(...): (line 395)
        # Processing the call arguments (line 395)
        
        
        # Call to ZerosPolesGain(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 's' (line 395)
        s_294586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 's', False)
        # Processing the call keyword arguments (line 395)
        kwargs_294587 = {}
        # Getting the type of 'ZerosPolesGain' (line 395)
        ZerosPolesGain_294585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 'ZerosPolesGain', False)
        # Calling ZerosPolesGain(args, kwargs) (line 395)
        ZerosPolesGain_call_result_294588 = invoke(stypy.reporting.localization.Localization(__file__, 395, 16), ZerosPolesGain_294585, *[s_294586], **kwargs_294587)
        
        # Getting the type of 's' (line 395)
        s_294589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 's', False)
        # Applying the binary operator 'isnot' (line 395)
        result_is_not_294590 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 16), 'isnot', ZerosPolesGain_call_result_294588, s_294589)
        
        # Processing the call keyword arguments (line 395)
        kwargs_294591 = {}
        # Getting the type of 'assert_' (line 395)
        assert__294584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 395)
        assert__call_result_294592 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__294584, *[result_is_not_294590], **kwargs_294591)
        
        
        # Call to assert_(...): (line 396)
        # Processing the call arguments (line 396)
        
        
        # Call to to_zpk(...): (line 396)
        # Processing the call keyword arguments (line 396)
        kwargs_294596 = {}
        # Getting the type of 's' (line 396)
        s_294594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 's', False)
        # Obtaining the member 'to_zpk' of a type (line 396)
        to_zpk_294595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), s_294594, 'to_zpk')
        # Calling to_zpk(args, kwargs) (line 396)
        to_zpk_call_result_294597 = invoke(stypy.reporting.localization.Localization(__file__, 396, 16), to_zpk_294595, *[], **kwargs_294596)
        
        # Getting the type of 's' (line 396)
        s_294598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 34), 's', False)
        # Applying the binary operator 'isnot' (line 396)
        result_is_not_294599 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 16), 'isnot', to_zpk_call_result_294597, s_294598)
        
        # Processing the call keyword arguments (line 396)
        kwargs_294600 = {}
        # Getting the type of 'assert_' (line 396)
        assert__294593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 396)
        assert__call_result_294601 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assert__294593, *[result_is_not_294599], **kwargs_294600)
        
        
        # ################# End of 'test_conversion(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_conversion' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_294602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294602)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_conversion'
        return stypy_return_type_294602


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 378, 0, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestZerosPolesGain.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestZerosPolesGain' (line 378)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 0), 'TestZerosPolesGain', TestZerosPolesGain)
# Declaration of the 'Test_dfreqresp' class

class Test_dfreqresp(object, ):

    @norecursion
    def test_manual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_manual'
        module_type_store = module_type_store.open_function_context('test_manual', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_manual')
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_manual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_manual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_manual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_manual(...)' code ##################

        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Assigning a Call to a Name (line 404):
        
        # Call to TransferFunction(...): (line 404)
        # Processing the call arguments (line 404)
        int_294604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 34), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 404)
        list_294605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 404)
        # Adding element type (line 404)
        int_294606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 37), list_294605, int_294606)
        # Adding element type (line 404)
        float_294607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 37), list_294605, float_294607)
        
        # Processing the call keyword arguments (line 404)
        float_294608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 51), 'float')
        keyword_294609 = float_294608
        kwargs_294610 = {'dt': keyword_294609}
        # Getting the type of 'TransferFunction' (line 404)
        TransferFunction_294603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 404)
        TransferFunction_call_result_294611 = invoke(stypy.reporting.localization.Localization(__file__, 404, 17), TransferFunction_294603, *[int_294604, list_294605], **kwargs_294610)
        
        # Assigning a type to the variable 'system' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'system', TransferFunction_call_result_294611)
        
        # Assigning a List to a Name (line 405):
        
        # Assigning a List to a Name (line 405):
        
        # Assigning a List to a Name (line 405):
        
        # Obtaining an instance of the builtin type 'list' (line 405)
        list_294612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 405)
        # Adding element type (line 405)
        float_294613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 12), list_294612, float_294613)
        # Adding element type (line 405)
        int_294614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 12), list_294612, int_294614)
        # Adding element type (line 405)
        int_294615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 405, 12), list_294612, int_294615)
        
        # Assigning a type to the variable 'w' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'w', list_294612)
        
        # Assigning a Call to a Tuple (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_294616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 8), 'int')
        
        # Call to dfreqresp(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'system' (line 406)
        system_294618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 25), 'system', False)
        # Processing the call keyword arguments (line 406)
        # Getting the type of 'w' (line 406)
        w_294619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'w', False)
        keyword_294620 = w_294619
        kwargs_294621 = {'w': keyword_294620}
        # Getting the type of 'dfreqresp' (line 406)
        dfreqresp_294617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 406)
        dfreqresp_call_result_294622 = invoke(stypy.reporting.localization.Localization(__file__, 406, 15), dfreqresp_294617, *[system_294618], **kwargs_294621)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___294623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), dfreqresp_call_result_294622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_294624 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), getitem___294623, int_294616)
        
        # Assigning a type to the variable 'tuple_var_assignment_292212' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_292212', subscript_call_result_294624)
        
        # Assigning a Subscript to a Name (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_294625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 8), 'int')
        
        # Call to dfreqresp(...): (line 406)
        # Processing the call arguments (line 406)
        # Getting the type of 'system' (line 406)
        system_294627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 25), 'system', False)
        # Processing the call keyword arguments (line 406)
        # Getting the type of 'w' (line 406)
        w_294628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'w', False)
        keyword_294629 = w_294628
        kwargs_294630 = {'w': keyword_294629}
        # Getting the type of 'dfreqresp' (line 406)
        dfreqresp_294626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 406)
        dfreqresp_call_result_294631 = invoke(stypy.reporting.localization.Localization(__file__, 406, 15), dfreqresp_294626, *[system_294627], **kwargs_294630)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___294632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), dfreqresp_call_result_294631, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_294633 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), getitem___294632, int_294625)
        
        # Assigning a type to the variable 'tuple_var_assignment_292213' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_292213', subscript_call_result_294633)
        
        # Assigning a Name to a Name (line 406):
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_292212' (line 406)
        tuple_var_assignment_292212_294634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_292212')
        # Assigning a type to the variable 'w' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'w', tuple_var_assignment_292212_294634)
        
        # Assigning a Name to a Name (line 406):
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_292213' (line 406)
        tuple_var_assignment_292213_294635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'tuple_var_assignment_292213')
        # Assigning a type to the variable 'H' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'H', tuple_var_assignment_292213_294635)
        
        # Assigning a List to a Name (line 409):
        
        # Assigning a List to a Name (line 409):
        
        # Assigning a List to a Name (line 409):
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_294636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        float_294637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_294636, float_294637)
        # Adding element type (line 409)
        float_294638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_294636, float_294638)
        # Adding element type (line 409)
        float_294639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 22), list_294636, float_294639)
        
        # Assigning a type to the variable 'expected_re' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'expected_re', list_294636)
        
        # Call to assert_almost_equal(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'H' (line 410)
        H_294641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'H', False)
        # Obtaining the member 'real' of a type (line 410)
        real_294642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 28), H_294641, 'real')
        # Getting the type of 'expected_re' (line 410)
        expected_re_294643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 'expected_re', False)
        # Processing the call keyword arguments (line 410)
        int_294644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 57), 'int')
        keyword_294645 = int_294644
        kwargs_294646 = {'decimal': keyword_294645}
        # Getting the type of 'assert_almost_equal' (line 410)
        assert_almost_equal_294640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 410)
        assert_almost_equal_call_result_294647 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), assert_almost_equal_294640, *[real_294642, expected_re_294643], **kwargs_294646)
        
        
        # Assigning a List to a Name (line 413):
        
        # Assigning a List to a Name (line 413):
        
        # Assigning a List to a Name (line 413):
        
        # Obtaining an instance of the builtin type 'list' (line 413)
        list_294648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 413)
        # Adding element type (line 413)
        float_294649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 22), list_294648, float_294649)
        # Adding element type (line 413)
        float_294650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 22), list_294648, float_294650)
        # Adding element type (line 413)
        float_294651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 22), list_294648, float_294651)
        
        # Assigning a type to the variable 'expected_im' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'expected_im', list_294648)
        
        # Call to assert_almost_equal(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'H' (line 414)
        H_294653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 28), 'H', False)
        # Obtaining the member 'imag' of a type (line 414)
        imag_294654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 28), H_294653, 'imag')
        # Getting the type of 'expected_im' (line 414)
        expected_im_294655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 36), 'expected_im', False)
        # Processing the call keyword arguments (line 414)
        int_294656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 57), 'int')
        keyword_294657 = int_294656
        kwargs_294658 = {'decimal': keyword_294657}
        # Getting the type of 'assert_almost_equal' (line 414)
        assert_almost_equal_294652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 414)
        assert_almost_equal_call_result_294659 = invoke(stypy.reporting.localization.Localization(__file__, 414, 8), assert_almost_equal_294652, *[imag_294654, expected_im_294655], **kwargs_294658)
        
        
        # ################# End of 'test_manual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_manual' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_294660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_manual'
        return stypy_return_type_294660


    @norecursion
    def test_auto(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_auto'
        module_type_store = module_type_store.open_function_context('test_auto', 416, 4, False)
        # Assigning a type to the variable 'self' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_auto')
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_auto.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_auto', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_auto', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_auto(...)' code ##################

        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Assigning a Call to a Name (line 419):
        
        # Call to TransferFunction(...): (line 419)
        # Processing the call arguments (line 419)
        int_294662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 34), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 419)
        list_294663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 419)
        # Adding element type (line 419)
        int_294664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 37), list_294663, int_294664)
        # Adding element type (line 419)
        float_294665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 37), list_294663, float_294665)
        
        # Processing the call keyword arguments (line 419)
        float_294666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 51), 'float')
        keyword_294667 = float_294666
        kwargs_294668 = {'dt': keyword_294667}
        # Getting the type of 'TransferFunction' (line 419)
        TransferFunction_294661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 419)
        TransferFunction_call_result_294669 = invoke(stypy.reporting.localization.Localization(__file__, 419, 17), TransferFunction_294661, *[int_294662, list_294663], **kwargs_294668)
        
        # Assigning a type to the variable 'system' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'system', TransferFunction_call_result_294669)
        
        # Assigning a List to a Name (line 420):
        
        # Assigning a List to a Name (line 420):
        
        # Assigning a List to a Name (line 420):
        
        # Obtaining an instance of the builtin type 'list' (line 420)
        list_294670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 420)
        # Adding element type (line 420)
        float_294671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), list_294670, float_294671)
        # Adding element type (line 420)
        int_294672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), list_294670, int_294672)
        # Adding element type (line 420)
        int_294673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), list_294670, int_294673)
        # Adding element type (line 420)
        int_294674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 12), list_294670, int_294674)
        
        # Assigning a type to the variable 'w' (line 420)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'w', list_294670)
        
        # Assigning a Call to a Tuple (line 421):
        
        # Assigning a Subscript to a Name (line 421):
        
        # Assigning a Subscript to a Name (line 421):
        
        # Obtaining the type of the subscript
        int_294675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 8), 'int')
        
        # Call to dfreqresp(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'system' (line 421)
        system_294677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'system', False)
        # Processing the call keyword arguments (line 421)
        # Getting the type of 'w' (line 421)
        w_294678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 35), 'w', False)
        keyword_294679 = w_294678
        kwargs_294680 = {'w': keyword_294679}
        # Getting the type of 'dfreqresp' (line 421)
        dfreqresp_294676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 421)
        dfreqresp_call_result_294681 = invoke(stypy.reporting.localization.Localization(__file__, 421, 15), dfreqresp_294676, *[system_294677], **kwargs_294680)
        
        # Obtaining the member '__getitem__' of a type (line 421)
        getitem___294682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), dfreqresp_call_result_294681, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 421)
        subscript_call_result_294683 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), getitem___294682, int_294675)
        
        # Assigning a type to the variable 'tuple_var_assignment_292214' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'tuple_var_assignment_292214', subscript_call_result_294683)
        
        # Assigning a Subscript to a Name (line 421):
        
        # Assigning a Subscript to a Name (line 421):
        
        # Obtaining the type of the subscript
        int_294684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 8), 'int')
        
        # Call to dfreqresp(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 'system' (line 421)
        system_294686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 25), 'system', False)
        # Processing the call keyword arguments (line 421)
        # Getting the type of 'w' (line 421)
        w_294687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 35), 'w', False)
        keyword_294688 = w_294687
        kwargs_294689 = {'w': keyword_294688}
        # Getting the type of 'dfreqresp' (line 421)
        dfreqresp_294685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 421)
        dfreqresp_call_result_294690 = invoke(stypy.reporting.localization.Localization(__file__, 421, 15), dfreqresp_294685, *[system_294686], **kwargs_294689)
        
        # Obtaining the member '__getitem__' of a type (line 421)
        getitem___294691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), dfreqresp_call_result_294690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 421)
        subscript_call_result_294692 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), getitem___294691, int_294684)
        
        # Assigning a type to the variable 'tuple_var_assignment_292215' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'tuple_var_assignment_292215', subscript_call_result_294692)
        
        # Assigning a Name to a Name (line 421):
        
        # Assigning a Name to a Name (line 421):
        # Getting the type of 'tuple_var_assignment_292214' (line 421)
        tuple_var_assignment_292214_294693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'tuple_var_assignment_292214')
        # Assigning a type to the variable 'w' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'w', tuple_var_assignment_292214_294693)
        
        # Assigning a Name to a Name (line 421):
        
        # Assigning a Name to a Name (line 421):
        # Getting the type of 'tuple_var_assignment_292215' (line 421)
        tuple_var_assignment_292215_294694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'tuple_var_assignment_292215')
        # Assigning a type to the variable 'H' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 11), 'H', tuple_var_assignment_292215_294694)
        
        # Assigning a Call to a Name (line 422):
        
        # Assigning a Call to a Name (line 422):
        
        # Assigning a Call to a Name (line 422):
        
        # Call to exp(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'w' (line 422)
        w_294697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'w', False)
        complex_294698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 24), 'complex')
        # Applying the binary operator '*' (line 422)
        result_mul_294699 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 20), '*', w_294697, complex_294698)
        
        # Processing the call keyword arguments (line 422)
        kwargs_294700 = {}
        # Getting the type of 'np' (line 422)
        np_294695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 13), 'np', False)
        # Obtaining the member 'exp' of a type (line 422)
        exp_294696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 13), np_294695, 'exp')
        # Calling exp(args, kwargs) (line 422)
        exp_call_result_294701 = invoke(stypy.reporting.localization.Localization(__file__, 422, 13), exp_294696, *[result_mul_294699], **kwargs_294700)
        
        # Assigning a type to the variable 'jw' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'jw', exp_call_result_294701)
        
        # Assigning a BinOp to a Name (line 423):
        
        # Assigning a BinOp to a Name (line 423):
        
        # Assigning a BinOp to a Name (line 423):
        
        # Call to polyval(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'system' (line 423)
        system_294704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), 'system', False)
        # Obtaining the member 'num' of a type (line 423)
        num_294705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 23), system_294704, 'num')
        # Getting the type of 'jw' (line 423)
        jw_294706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 35), 'jw', False)
        # Processing the call keyword arguments (line 423)
        kwargs_294707 = {}
        # Getting the type of 'np' (line 423)
        np_294702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'np', False)
        # Obtaining the member 'polyval' of a type (line 423)
        polyval_294703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), np_294702, 'polyval')
        # Calling polyval(args, kwargs) (line 423)
        polyval_call_result_294708 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), polyval_294703, *[num_294705, jw_294706], **kwargs_294707)
        
        
        # Call to polyval(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'system' (line 423)
        system_294711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 52), 'system', False)
        # Obtaining the member 'den' of a type (line 423)
        den_294712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 52), system_294711, 'den')
        # Getting the type of 'jw' (line 423)
        jw_294713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 64), 'jw', False)
        # Processing the call keyword arguments (line 423)
        kwargs_294714 = {}
        # Getting the type of 'np' (line 423)
        np_294709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 41), 'np', False)
        # Obtaining the member 'polyval' of a type (line 423)
        polyval_294710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 41), np_294709, 'polyval')
        # Calling polyval(args, kwargs) (line 423)
        polyval_call_result_294715 = invoke(stypy.reporting.localization.Localization(__file__, 423, 41), polyval_294710, *[den_294712, jw_294713], **kwargs_294714)
        
        # Applying the binary operator 'div' (line 423)
        result_div_294716 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 12), 'div', polyval_call_result_294708, polyval_call_result_294715)
        
        # Assigning a type to the variable 'y' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'y', result_div_294716)
        
        # Assigning a Attribute to a Name (line 426):
        
        # Assigning a Attribute to a Name (line 426):
        
        # Assigning a Attribute to a Name (line 426):
        # Getting the type of 'y' (line 426)
        y_294717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 22), 'y')
        # Obtaining the member 'real' of a type (line 426)
        real_294718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 22), y_294717, 'real')
        # Assigning a type to the variable 'expected_re' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'expected_re', real_294718)
        
        # Call to assert_almost_equal(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'H' (line 427)
        H_294720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 28), 'H', False)
        # Obtaining the member 'real' of a type (line 427)
        real_294721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 28), H_294720, 'real')
        # Getting the type of 'expected_re' (line 427)
        expected_re_294722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 36), 'expected_re', False)
        # Processing the call keyword arguments (line 427)
        kwargs_294723 = {}
        # Getting the type of 'assert_almost_equal' (line 427)
        assert_almost_equal_294719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 427)
        assert_almost_equal_call_result_294724 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), assert_almost_equal_294719, *[real_294721, expected_re_294722], **kwargs_294723)
        
        
        # Assigning a Attribute to a Name (line 430):
        
        # Assigning a Attribute to a Name (line 430):
        
        # Assigning a Attribute to a Name (line 430):
        # Getting the type of 'y' (line 430)
        y_294725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 22), 'y')
        # Obtaining the member 'imag' of a type (line 430)
        imag_294726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 22), y_294725, 'imag')
        # Assigning a type to the variable 'expected_im' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'expected_im', imag_294726)
        
        # Call to assert_almost_equal(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'H' (line 431)
        H_294728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'H', False)
        # Obtaining the member 'imag' of a type (line 431)
        imag_294729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 28), H_294728, 'imag')
        # Getting the type of 'expected_im' (line 431)
        expected_im_294730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 36), 'expected_im', False)
        # Processing the call keyword arguments (line 431)
        kwargs_294731 = {}
        # Getting the type of 'assert_almost_equal' (line 431)
        assert_almost_equal_294727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 431)
        assert_almost_equal_call_result_294732 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), assert_almost_equal_294727, *[imag_294729, expected_im_294730], **kwargs_294731)
        
        
        # ################# End of 'test_auto(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_auto' in the type store
        # Getting the type of 'stypy_return_type' (line 416)
        stypy_return_type_294733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_auto'
        return stypy_return_type_294733


    @norecursion
    def test_freq_range(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_freq_range'
        module_type_store = module_type_store.open_function_context('test_freq_range', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_freq_range')
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_freq_range.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_freq_range', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_freq_range', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_freq_range(...)' code ##################

        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Assigning a Call to a Name (line 437):
        
        # Call to TransferFunction(...): (line 437)
        # Processing the call arguments (line 437)
        int_294735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 34), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 437)
        list_294736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 437)
        # Adding element type (line 437)
        int_294737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 37), list_294736, int_294737)
        # Adding element type (line 437)
        float_294738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 37), list_294736, float_294738)
        
        # Processing the call keyword arguments (line 437)
        float_294739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 51), 'float')
        keyword_294740 = float_294739
        kwargs_294741 = {'dt': keyword_294740}
        # Getting the type of 'TransferFunction' (line 437)
        TransferFunction_294734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 437)
        TransferFunction_call_result_294742 = invoke(stypy.reporting.localization.Localization(__file__, 437, 17), TransferFunction_294734, *[int_294735, list_294736], **kwargs_294741)
        
        # Assigning a type to the variable 'system' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'system', TransferFunction_call_result_294742)
        
        # Assigning a Num to a Name (line 438):
        
        # Assigning a Num to a Name (line 438):
        
        # Assigning a Num to a Name (line 438):
        int_294743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 12), 'int')
        # Assigning a type to the variable 'n' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'n', int_294743)
        
        # Assigning a Call to a Name (line 439):
        
        # Assigning a Call to a Name (line 439):
        
        # Assigning a Call to a Name (line 439):
        
        # Call to linspace(...): (line 439)
        # Processing the call arguments (line 439)
        int_294746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 33), 'int')
        # Getting the type of 'np' (line 439)
        np_294747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 36), 'np', False)
        # Obtaining the member 'pi' of a type (line 439)
        pi_294748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 36), np_294747, 'pi')
        int_294749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 43), 'int')
        # Processing the call keyword arguments (line 439)
        # Getting the type of 'False' (line 439)
        False_294750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 56), 'False', False)
        keyword_294751 = False_294750
        kwargs_294752 = {'endpoint': keyword_294751}
        # Getting the type of 'np' (line 439)
        np_294744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'np', False)
        # Obtaining the member 'linspace' of a type (line 439)
        linspace_294745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 21), np_294744, 'linspace')
        # Calling linspace(args, kwargs) (line 439)
        linspace_call_result_294753 = invoke(stypy.reporting.localization.Localization(__file__, 439, 21), linspace_294745, *[int_294746, pi_294748, int_294749], **kwargs_294752)
        
        # Assigning a type to the variable 'expected_w' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'expected_w', linspace_call_result_294753)
        
        # Assigning a Call to a Tuple (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_294754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 8), 'int')
        
        # Call to dfreqresp(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'system' (line 440)
        system_294756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'system', False)
        # Processing the call keyword arguments (line 440)
        # Getting the type of 'n' (line 440)
        n_294757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 35), 'n', False)
        keyword_294758 = n_294757
        kwargs_294759 = {'n': keyword_294758}
        # Getting the type of 'dfreqresp' (line 440)
        dfreqresp_294755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 440)
        dfreqresp_call_result_294760 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), dfreqresp_294755, *[system_294756], **kwargs_294759)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___294761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), dfreqresp_call_result_294760, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_294762 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), getitem___294761, int_294754)
        
        # Assigning a type to the variable 'tuple_var_assignment_292216' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'tuple_var_assignment_292216', subscript_call_result_294762)
        
        # Assigning a Subscript to a Name (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        int_294763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 8), 'int')
        
        # Call to dfreqresp(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'system' (line 440)
        system_294765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'system', False)
        # Processing the call keyword arguments (line 440)
        # Getting the type of 'n' (line 440)
        n_294766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 35), 'n', False)
        keyword_294767 = n_294766
        kwargs_294768 = {'n': keyword_294767}
        # Getting the type of 'dfreqresp' (line 440)
        dfreqresp_294764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 15), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 440)
        dfreqresp_call_result_294769 = invoke(stypy.reporting.localization.Localization(__file__, 440, 15), dfreqresp_294764, *[system_294765], **kwargs_294768)
        
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___294770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 8), dfreqresp_call_result_294769, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_294771 = invoke(stypy.reporting.localization.Localization(__file__, 440, 8), getitem___294770, int_294763)
        
        # Assigning a type to the variable 'tuple_var_assignment_292217' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'tuple_var_assignment_292217', subscript_call_result_294771)
        
        # Assigning a Name to a Name (line 440):
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_292216' (line 440)
        tuple_var_assignment_292216_294772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'tuple_var_assignment_292216')
        # Assigning a type to the variable 'w' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'w', tuple_var_assignment_292216_294772)
        
        # Assigning a Name to a Name (line 440):
        
        # Assigning a Name to a Name (line 440):
        # Getting the type of 'tuple_var_assignment_292217' (line 440)
        tuple_var_assignment_292217_294773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'tuple_var_assignment_292217')
        # Assigning a type to the variable 'H' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'H', tuple_var_assignment_292217_294773)
        
        # Call to assert_almost_equal(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'w' (line 441)
        w_294775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'w', False)
        # Getting the type of 'expected_w' (line 441)
        expected_w_294776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 31), 'expected_w', False)
        # Processing the call keyword arguments (line 441)
        kwargs_294777 = {}
        # Getting the type of 'assert_almost_equal' (line 441)
        assert_almost_equal_294774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 441)
        assert_almost_equal_call_result_294778 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), assert_almost_equal_294774, *[w_294775, expected_w_294776], **kwargs_294777)
        
        
        # ################# End of 'test_freq_range(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_freq_range' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_294779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294779)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_freq_range'
        return stypy_return_type_294779


    @norecursion
    def test_pole_one(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pole_one'
        module_type_store = module_type_store.open_function_context('test_pole_one', 443, 4, False)
        # Assigning a type to the variable 'self' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_pole_one')
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_pole_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_pole_one', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pole_one', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pole_one(...)' code ##################

        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Assigning a Call to a Name (line 446):
        
        # Call to TransferFunction(...): (line 446)
        # Processing the call arguments (line 446)
        
        # Obtaining an instance of the builtin type 'list' (line 446)
        list_294781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 446)
        # Adding element type (line 446)
        int_294782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 34), list_294781, int_294782)
        
        
        # Obtaining an instance of the builtin type 'list' (line 446)
        list_294783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 446)
        # Adding element type (line 446)
        int_294784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 39), list_294783, int_294784)
        # Adding element type (line 446)
        int_294785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 39), list_294783, int_294785)
        
        # Processing the call keyword arguments (line 446)
        float_294786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 51), 'float')
        keyword_294787 = float_294786
        kwargs_294788 = {'dt': keyword_294787}
        # Getting the type of 'TransferFunction' (line 446)
        TransferFunction_294780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 446)
        TransferFunction_call_result_294789 = invoke(stypy.reporting.localization.Localization(__file__, 446, 17), TransferFunction_294780, *[list_294781, list_294783], **kwargs_294788)
        
        # Assigning a type to the variable 'system' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'system', TransferFunction_call_result_294789)
        
        # Call to suppress_warnings(...): (line 448)
        # Processing the call keyword arguments (line 448)
        kwargs_294791 = {}
        # Getting the type of 'suppress_warnings' (line 448)
        suppress_warnings_294790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 448)
        suppress_warnings_call_result_294792 = invoke(stypy.reporting.localization.Localization(__file__, 448, 13), suppress_warnings_294790, *[], **kwargs_294791)
        
        with_294793 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 448, 13), suppress_warnings_call_result_294792, 'with parameter', '__enter__', '__exit__')

        if with_294793:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 448)
            enter___294794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 13), suppress_warnings_call_result_294792, '__enter__')
            with_enter_294795 = invoke(stypy.reporting.localization.Localization(__file__, 448, 13), enter___294794)
            # Assigning a type to the variable 'sup' (line 448)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 13), 'sup', with_enter_294795)
            
            # Call to filter(...): (line 449)
            # Processing the call arguments (line 449)
            # Getting the type of 'RuntimeWarning' (line 449)
            RuntimeWarning_294798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 23), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 449)
            str_294799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 47), 'str', 'divide by zero')
            keyword_294800 = str_294799
            kwargs_294801 = {'message': keyword_294800}
            # Getting the type of 'sup' (line 449)
            sup_294796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 449)
            filter_294797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 12), sup_294796, 'filter')
            # Calling filter(args, kwargs) (line 449)
            filter_call_result_294802 = invoke(stypy.reporting.localization.Localization(__file__, 449, 12), filter_294797, *[RuntimeWarning_294798], **kwargs_294801)
            
            
            # Call to filter(...): (line 450)
            # Processing the call arguments (line 450)
            # Getting the type of 'RuntimeWarning' (line 450)
            RuntimeWarning_294805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 23), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 450)
            str_294806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 47), 'str', 'invalid value encountered')
            keyword_294807 = str_294806
            kwargs_294808 = {'message': keyword_294807}
            # Getting the type of 'sup' (line 450)
            sup_294803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 450)
            filter_294804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 12), sup_294803, 'filter')
            # Calling filter(args, kwargs) (line 450)
            filter_call_result_294809 = invoke(stypy.reporting.localization.Localization(__file__, 450, 12), filter_294804, *[RuntimeWarning_294805], **kwargs_294808)
            
            
            # Assigning a Call to a Tuple (line 451):
            
            # Assigning a Subscript to a Name (line 451):
            
            # Assigning a Subscript to a Name (line 451):
            
            # Obtaining the type of the subscript
            int_294810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 12), 'int')
            
            # Call to dfreqresp(...): (line 451)
            # Processing the call arguments (line 451)
            # Getting the type of 'system' (line 451)
            system_294812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 29), 'system', False)
            # Processing the call keyword arguments (line 451)
            int_294813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'int')
            keyword_294814 = int_294813
            kwargs_294815 = {'n': keyword_294814}
            # Getting the type of 'dfreqresp' (line 451)
            dfreqresp_294811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 19), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 451)
            dfreqresp_call_result_294816 = invoke(stypy.reporting.localization.Localization(__file__, 451, 19), dfreqresp_294811, *[system_294812], **kwargs_294815)
            
            # Obtaining the member '__getitem__' of a type (line 451)
            getitem___294817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), dfreqresp_call_result_294816, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 451)
            subscript_call_result_294818 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___294817, int_294810)
            
            # Assigning a type to the variable 'tuple_var_assignment_292218' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_292218', subscript_call_result_294818)
            
            # Assigning a Subscript to a Name (line 451):
            
            # Assigning a Subscript to a Name (line 451):
            
            # Obtaining the type of the subscript
            int_294819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 12), 'int')
            
            # Call to dfreqresp(...): (line 451)
            # Processing the call arguments (line 451)
            # Getting the type of 'system' (line 451)
            system_294821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 29), 'system', False)
            # Processing the call keyword arguments (line 451)
            int_294822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 39), 'int')
            keyword_294823 = int_294822
            kwargs_294824 = {'n': keyword_294823}
            # Getting the type of 'dfreqresp' (line 451)
            dfreqresp_294820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 19), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 451)
            dfreqresp_call_result_294825 = invoke(stypy.reporting.localization.Localization(__file__, 451, 19), dfreqresp_294820, *[system_294821], **kwargs_294824)
            
            # Obtaining the member '__getitem__' of a type (line 451)
            getitem___294826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 12), dfreqresp_call_result_294825, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 451)
            subscript_call_result_294827 = invoke(stypy.reporting.localization.Localization(__file__, 451, 12), getitem___294826, int_294819)
            
            # Assigning a type to the variable 'tuple_var_assignment_292219' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_292219', subscript_call_result_294827)
            
            # Assigning a Name to a Name (line 451):
            
            # Assigning a Name to a Name (line 451):
            # Getting the type of 'tuple_var_assignment_292218' (line 451)
            tuple_var_assignment_292218_294828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_292218')
            # Assigning a type to the variable 'w' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'w', tuple_var_assignment_292218_294828)
            
            # Assigning a Name to a Name (line 451):
            
            # Assigning a Name to a Name (line 451):
            # Getting the type of 'tuple_var_assignment_292219' (line 451)
            tuple_var_assignment_292219_294829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'tuple_var_assignment_292219')
            # Assigning a type to the variable 'H' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 15), 'H', tuple_var_assignment_292219_294829)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 448)
            exit___294830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 13), suppress_warnings_call_result_294792, '__exit__')
            with_exit_294831 = invoke(stypy.reporting.localization.Localization(__file__, 448, 13), exit___294830, None, None, None)

        
        # Call to assert_equal(...): (line 452)
        # Processing the call arguments (line 452)
        
        # Obtaining the type of the subscript
        int_294833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 23), 'int')
        # Getting the type of 'w' (line 452)
        w_294834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 21), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 452)
        getitem___294835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 21), w_294834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 452)
        subscript_call_result_294836 = invoke(stypy.reporting.localization.Localization(__file__, 452, 21), getitem___294835, int_294833)
        
        float_294837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 27), 'float')
        # Processing the call keyword arguments (line 452)
        kwargs_294838 = {}
        # Getting the type of 'assert_equal' (line 452)
        assert_equal_294832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 452)
        assert_equal_call_result_294839 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), assert_equal_294832, *[subscript_call_result_294836, float_294837], **kwargs_294838)
        
        
        # ################# End of 'test_pole_one(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pole_one' in the type store
        # Getting the type of 'stypy_return_type' (line 443)
        stypy_return_type_294840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pole_one'
        return stypy_return_type_294840


    @norecursion
    def test_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error'
        module_type_store = module_type_store.open_function_context('test_error', 454, 4, False)
        # Assigning a type to the variable 'self' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_error')
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_error.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error(...)' code ##################

        
        # Assigning a Call to a Name (line 456):
        
        # Assigning a Call to a Name (line 456):
        
        # Assigning a Call to a Name (line 456):
        
        # Call to lti(...): (line 456)
        # Processing the call arguments (line 456)
        
        # Obtaining an instance of the builtin type 'list' (line 456)
        list_294842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 456)
        # Adding element type (line 456)
        int_294843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 21), list_294842, int_294843)
        
        
        # Obtaining an instance of the builtin type 'list' (line 456)
        list_294844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 456)
        # Adding element type (line 456)
        int_294845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 26), list_294844, int_294845)
        # Adding element type (line 456)
        int_294846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 456, 26), list_294844, int_294846)
        
        # Processing the call keyword arguments (line 456)
        kwargs_294847 = {}
        # Getting the type of 'lti' (line 456)
        lti_294841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'lti', False)
        # Calling lti(args, kwargs) (line 456)
        lti_call_result_294848 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), lti_294841, *[list_294842, list_294844], **kwargs_294847)
        
        # Assigning a type to the variable 'system' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'system', lti_call_result_294848)
        
        # Call to assert_raises(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'AttributeError' (line 457)
        AttributeError_294850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 22), 'AttributeError', False)
        # Getting the type of 'dfreqresp' (line 457)
        dfreqresp_294851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 38), 'dfreqresp', False)
        # Getting the type of 'system' (line 457)
        system_294852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'system', False)
        # Processing the call keyword arguments (line 457)
        kwargs_294853 = {}
        # Getting the type of 'assert_raises' (line 457)
        assert_raises_294849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 457)
        assert_raises_call_result_294854 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), assert_raises_294849, *[AttributeError_294850, dfreqresp_294851, system_294852], **kwargs_294853)
        
        
        # ################# End of 'test_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error' in the type store
        # Getting the type of 'stypy_return_type' (line 454)
        stypy_return_type_294855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error'
        return stypy_return_type_294855


    @norecursion
    def test_from_state_space(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_from_state_space'
        module_type_store = module_type_store.open_function_context('test_from_state_space', 459, 4, False)
        # Assigning a type to the variable 'self' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_from_state_space')
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_from_state_space.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_from_state_space', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_from_state_space', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_from_state_space(...)' code ##################

        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Assigning a Call to a Name (line 462):
        
        # Call to dlti(...): (line 462)
        # Processing the call arguments (line 462)
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_294857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        int_294858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 25), list_294857, int_294858)
        
        
        # Obtaining an instance of the builtin type 'list' (line 462)
        list_294859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 462)
        # Adding element type (line 462)
        int_294860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 30), list_294859, int_294860)
        # Adding element type (line 462)
        float_294861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 30), list_294859, float_294861)
        # Adding element type (line 462)
        int_294862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 30), list_294859, int_294862)
        # Adding element type (line 462)
        int_294863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 30), list_294859, int_294863)
        
        # Processing the call keyword arguments (line 462)
        kwargs_294864 = {}
        # Getting the type of 'dlti' (line 462)
        dlti_294856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'dlti', False)
        # Calling dlti(args, kwargs) (line 462)
        dlti_call_result_294865 = invoke(stypy.reporting.localization.Localization(__file__, 462, 20), dlti_294856, *[list_294857, list_294859], **kwargs_294864)
        
        # Assigning a type to the variable 'system_TF' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'system_TF', dlti_call_result_294865)
        
        # Assigning a Call to a Name (line 464):
        
        # Assigning a Call to a Name (line 464):
        
        # Assigning a Call to a Name (line 464):
        
        # Call to array(...): (line 464)
        # Processing the call arguments (line 464)
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_294868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        # Adding element type (line 464)
        
        # Obtaining an instance of the builtin type 'list' (line 464)
        list_294869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 464)
        # Adding element type (line 464)
        float_294870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 22), list_294869, float_294870)
        # Adding element type (line 464)
        int_294871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 22), list_294869, int_294871)
        # Adding element type (line 464)
        int_294872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 22), list_294869, int_294872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 21), list_294868, list_294869)
        # Adding element type (line 464)
        
        # Obtaining an instance of the builtin type 'list' (line 465)
        list_294873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 465)
        # Adding element type (line 465)
        int_294874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 22), list_294873, int_294874)
        # Adding element type (line 465)
        int_294875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 22), list_294873, int_294875)
        # Adding element type (line 465)
        int_294876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 22), list_294873, int_294876)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 21), list_294868, list_294873)
        # Adding element type (line 464)
        
        # Obtaining an instance of the builtin type 'list' (line 466)
        list_294877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 466)
        # Adding element type (line 466)
        int_294878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 22), list_294877, int_294878)
        # Adding element type (line 466)
        int_294879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 22), list_294877, int_294879)
        # Adding element type (line 466)
        int_294880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 22), list_294877, int_294880)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 21), list_294868, list_294877)
        
        # Processing the call keyword arguments (line 464)
        kwargs_294881 = {}
        # Getting the type of 'np' (line 464)
        np_294866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 464)
        array_294867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 12), np_294866, 'array')
        # Calling array(args, kwargs) (line 464)
        array_call_result_294882 = invoke(stypy.reporting.localization.Localization(__file__, 464, 12), array_294867, *[list_294868], **kwargs_294881)
        
        # Assigning a type to the variable 'A' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'A', array_call_result_294882)
        
        # Assigning a Attribute to a Name (line 467):
        
        # Assigning a Attribute to a Name (line 467):
        
        # Assigning a Attribute to a Name (line 467):
        
        # Call to array(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Obtaining an instance of the builtin type 'list' (line 467)
        list_294885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 467)
        # Adding element type (line 467)
        
        # Obtaining an instance of the builtin type 'list' (line 467)
        list_294886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 467)
        # Adding element type (line 467)
        int_294887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 22), list_294886, int_294887)
        # Adding element type (line 467)
        int_294888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 22), list_294886, int_294888)
        # Adding element type (line 467)
        int_294889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 22), list_294886, int_294889)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 21), list_294885, list_294886)
        
        # Processing the call keyword arguments (line 467)
        kwargs_294890 = {}
        # Getting the type of 'np' (line 467)
        np_294883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 467)
        array_294884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), np_294883, 'array')
        # Calling array(args, kwargs) (line 467)
        array_call_result_294891 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), array_294884, *[list_294885], **kwargs_294890)
        
        # Obtaining the member 'T' of a type (line 467)
        T_294892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 12), array_call_result_294891, 'T')
        # Assigning a type to the variable 'B' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'B', T_294892)
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Call to array(...): (line 468)
        # Processing the call arguments (line 468)
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_294895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        # Adding element type (line 468)
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_294896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        # Adding element type (line 468)
        int_294897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 22), list_294896, int_294897)
        # Adding element type (line 468)
        int_294898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 22), list_294896, int_294898)
        # Adding element type (line 468)
        int_294899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 22), list_294896, int_294899)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 21), list_294895, list_294896)
        
        # Processing the call keyword arguments (line 468)
        kwargs_294900 = {}
        # Getting the type of 'np' (line 468)
        np_294893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 468)
        array_294894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 12), np_294893, 'array')
        # Calling array(args, kwargs) (line 468)
        array_call_result_294901 = invoke(stypy.reporting.localization.Localization(__file__, 468, 12), array_294894, *[list_294895], **kwargs_294900)
        
        # Assigning a type to the variable 'C' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'C', array_call_result_294901)
        
        # Assigning a Num to a Name (line 469):
        
        # Assigning a Num to a Name (line 469):
        
        # Assigning a Num to a Name (line 469):
        int_294902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 12), 'int')
        # Assigning a type to the variable 'D' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'D', int_294902)
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to dlti(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'A' (line 471)
        A_294904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 25), 'A', False)
        # Getting the type of 'B' (line 471)
        B_294905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), 'B', False)
        # Getting the type of 'C' (line 471)
        C_294906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 31), 'C', False)
        # Getting the type of 'D' (line 471)
        D_294907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 34), 'D', False)
        # Processing the call keyword arguments (line 471)
        kwargs_294908 = {}
        # Getting the type of 'dlti' (line 471)
        dlti_294903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'dlti', False)
        # Calling dlti(args, kwargs) (line 471)
        dlti_call_result_294909 = invoke(stypy.reporting.localization.Localization(__file__, 471, 20), dlti_294903, *[A_294904, B_294905, C_294906, D_294907], **kwargs_294908)
        
        # Assigning a type to the variable 'system_SS' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'system_SS', dlti_call_result_294909)
        
        # Assigning a BinOp to a Name (line 472):
        
        # Assigning a BinOp to a Name (line 472):
        
        # Assigning a BinOp to a Name (line 472):
        float_294910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 12), 'float')
        
        # Call to arange(...): (line 472)
        # Processing the call arguments (line 472)
        int_294913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 28), 'int')
        int_294914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 31), 'int')
        float_294915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 33), 'float')
        # Processing the call keyword arguments (line 472)
        kwargs_294916 = {}
        # Getting the type of 'np' (line 472)
        np_294911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 18), 'np', False)
        # Obtaining the member 'arange' of a type (line 472)
        arange_294912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 18), np_294911, 'arange')
        # Calling arange(args, kwargs) (line 472)
        arange_call_result_294917 = invoke(stypy.reporting.localization.Localization(__file__, 472, 18), arange_294912, *[int_294913, int_294914, float_294915], **kwargs_294916)
        
        # Applying the binary operator '**' (line 472)
        result_pow_294918 = python_operator(stypy.reporting.localization.Localization(__file__, 472, 12), '**', float_294910, arange_call_result_294917)
        
        # Assigning a type to the variable 'w' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'w', result_pow_294918)
        
        # Call to suppress_warnings(...): (line 473)
        # Processing the call keyword arguments (line 473)
        kwargs_294920 = {}
        # Getting the type of 'suppress_warnings' (line 473)
        suppress_warnings_294919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 473)
        suppress_warnings_call_result_294921 = invoke(stypy.reporting.localization.Localization(__file__, 473, 13), suppress_warnings_294919, *[], **kwargs_294920)
        
        with_294922 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 473, 13), suppress_warnings_call_result_294921, 'with parameter', '__enter__', '__exit__')

        if with_294922:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 473)
            enter___294923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 13), suppress_warnings_call_result_294921, '__enter__')
            with_enter_294924 = invoke(stypy.reporting.localization.Localization(__file__, 473, 13), enter___294923)
            # Assigning a type to the variable 'sup' (line 473)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 13), 'sup', with_enter_294924)
            
            # Call to filter(...): (line 474)
            # Processing the call arguments (line 474)
            # Getting the type of 'BadCoefficients' (line 474)
            BadCoefficients_294927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'BadCoefficients', False)
            # Processing the call keyword arguments (line 474)
            kwargs_294928 = {}
            # Getting the type of 'sup' (line 474)
            sup_294925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 474)
            filter_294926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 12), sup_294925, 'filter')
            # Calling filter(args, kwargs) (line 474)
            filter_call_result_294929 = invoke(stypy.reporting.localization.Localization(__file__, 474, 12), filter_294926, *[BadCoefficients_294927], **kwargs_294928)
            
            
            # Assigning a Call to a Tuple (line 475):
            
            # Assigning a Subscript to a Name (line 475):
            
            # Assigning a Subscript to a Name (line 475):
            
            # Obtaining the type of the subscript
            int_294930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
            
            # Call to dfreqresp(...): (line 475)
            # Processing the call arguments (line 475)
            # Getting the type of 'system_TF' (line 475)
            system_TF_294932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'system_TF', False)
            # Processing the call keyword arguments (line 475)
            # Getting the type of 'w' (line 475)
            w_294933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 44), 'w', False)
            keyword_294934 = w_294933
            kwargs_294935 = {'w': keyword_294934}
            # Getting the type of 'dfreqresp' (line 475)
            dfreqresp_294931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 475)
            dfreqresp_call_result_294936 = invoke(stypy.reporting.localization.Localization(__file__, 475, 21), dfreqresp_294931, *[system_TF_294932], **kwargs_294935)
            
            # Obtaining the member '__getitem__' of a type (line 475)
            getitem___294937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), dfreqresp_call_result_294936, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 475)
            subscript_call_result_294938 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___294937, int_294930)
            
            # Assigning a type to the variable 'tuple_var_assignment_292220' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_292220', subscript_call_result_294938)
            
            # Assigning a Subscript to a Name (line 475):
            
            # Assigning a Subscript to a Name (line 475):
            
            # Obtaining the type of the subscript
            int_294939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 12), 'int')
            
            # Call to dfreqresp(...): (line 475)
            # Processing the call arguments (line 475)
            # Getting the type of 'system_TF' (line 475)
            system_TF_294941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 31), 'system_TF', False)
            # Processing the call keyword arguments (line 475)
            # Getting the type of 'w' (line 475)
            w_294942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 44), 'w', False)
            keyword_294943 = w_294942
            kwargs_294944 = {'w': keyword_294943}
            # Getting the type of 'dfreqresp' (line 475)
            dfreqresp_294940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 21), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 475)
            dfreqresp_call_result_294945 = invoke(stypy.reporting.localization.Localization(__file__, 475, 21), dfreqresp_294940, *[system_TF_294941], **kwargs_294944)
            
            # Obtaining the member '__getitem__' of a type (line 475)
            getitem___294946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), dfreqresp_call_result_294945, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 475)
            subscript_call_result_294947 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), getitem___294946, int_294939)
            
            # Assigning a type to the variable 'tuple_var_assignment_292221' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_292221', subscript_call_result_294947)
            
            # Assigning a Name to a Name (line 475):
            
            # Assigning a Name to a Name (line 475):
            # Getting the type of 'tuple_var_assignment_292220' (line 475)
            tuple_var_assignment_292220_294948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_292220')
            # Assigning a type to the variable 'w1' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'w1', tuple_var_assignment_292220_294948)
            
            # Assigning a Name to a Name (line 475):
            
            # Assigning a Name to a Name (line 475):
            # Getting the type of 'tuple_var_assignment_292221' (line 475)
            tuple_var_assignment_292221_294949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'tuple_var_assignment_292221')
            # Assigning a type to the variable 'H1' (line 475)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 16), 'H1', tuple_var_assignment_292221_294949)
            
            # Assigning a Call to a Tuple (line 476):
            
            # Assigning a Subscript to a Name (line 476):
            
            # Assigning a Subscript to a Name (line 476):
            
            # Obtaining the type of the subscript
            int_294950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 12), 'int')
            
            # Call to dfreqresp(...): (line 476)
            # Processing the call arguments (line 476)
            # Getting the type of 'system_SS' (line 476)
            system_SS_294952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 31), 'system_SS', False)
            # Processing the call keyword arguments (line 476)
            # Getting the type of 'w' (line 476)
            w_294953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 44), 'w', False)
            keyword_294954 = w_294953
            kwargs_294955 = {'w': keyword_294954}
            # Getting the type of 'dfreqresp' (line 476)
            dfreqresp_294951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 21), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 476)
            dfreqresp_call_result_294956 = invoke(stypy.reporting.localization.Localization(__file__, 476, 21), dfreqresp_294951, *[system_SS_294952], **kwargs_294955)
            
            # Obtaining the member '__getitem__' of a type (line 476)
            getitem___294957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), dfreqresp_call_result_294956, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 476)
            subscript_call_result_294958 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), getitem___294957, int_294950)
            
            # Assigning a type to the variable 'tuple_var_assignment_292222' (line 476)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'tuple_var_assignment_292222', subscript_call_result_294958)
            
            # Assigning a Subscript to a Name (line 476):
            
            # Assigning a Subscript to a Name (line 476):
            
            # Obtaining the type of the subscript
            int_294959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 12), 'int')
            
            # Call to dfreqresp(...): (line 476)
            # Processing the call arguments (line 476)
            # Getting the type of 'system_SS' (line 476)
            system_SS_294961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 31), 'system_SS', False)
            # Processing the call keyword arguments (line 476)
            # Getting the type of 'w' (line 476)
            w_294962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 44), 'w', False)
            keyword_294963 = w_294962
            kwargs_294964 = {'w': keyword_294963}
            # Getting the type of 'dfreqresp' (line 476)
            dfreqresp_294960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 21), 'dfreqresp', False)
            # Calling dfreqresp(args, kwargs) (line 476)
            dfreqresp_call_result_294965 = invoke(stypy.reporting.localization.Localization(__file__, 476, 21), dfreqresp_294960, *[system_SS_294961], **kwargs_294964)
            
            # Obtaining the member '__getitem__' of a type (line 476)
            getitem___294966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), dfreqresp_call_result_294965, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 476)
            subscript_call_result_294967 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), getitem___294966, int_294959)
            
            # Assigning a type to the variable 'tuple_var_assignment_292223' (line 476)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'tuple_var_assignment_292223', subscript_call_result_294967)
            
            # Assigning a Name to a Name (line 476):
            
            # Assigning a Name to a Name (line 476):
            # Getting the type of 'tuple_var_assignment_292222' (line 476)
            tuple_var_assignment_292222_294968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'tuple_var_assignment_292222')
            # Assigning a type to the variable 'w2' (line 476)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'w2', tuple_var_assignment_292222_294968)
            
            # Assigning a Name to a Name (line 476):
            
            # Assigning a Name to a Name (line 476):
            # Getting the type of 'tuple_var_assignment_292223' (line 476)
            tuple_var_assignment_292223_294969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'tuple_var_assignment_292223')
            # Assigning a type to the variable 'H2' (line 476)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'H2', tuple_var_assignment_292223_294969)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 473)
            exit___294970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 13), suppress_warnings_call_result_294921, '__exit__')
            with_exit_294971 = invoke(stypy.reporting.localization.Localization(__file__, 473, 13), exit___294970, None, None, None)

        
        # Call to assert_almost_equal(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'H1' (line 478)
        H1_294973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 28), 'H1', False)
        # Getting the type of 'H2' (line 478)
        H2_294974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 32), 'H2', False)
        # Processing the call keyword arguments (line 478)
        kwargs_294975 = {}
        # Getting the type of 'assert_almost_equal' (line 478)
        assert_almost_equal_294972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 478)
        assert_almost_equal_call_result_294976 = invoke(stypy.reporting.localization.Localization(__file__, 478, 8), assert_almost_equal_294972, *[H1_294973, H2_294974], **kwargs_294975)
        
        
        # ################# End of 'test_from_state_space(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_from_state_space' in the type store
        # Getting the type of 'stypy_return_type' (line 459)
        stypy_return_type_294977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294977)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_from_state_space'
        return stypy_return_type_294977


    @norecursion
    def test_from_zpk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_from_zpk'
        module_type_store = module_type_store.open_function_context('test_from_zpk', 480, 4, False)
        # Assigning a type to the variable 'self' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_localization', localization)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_function_name', 'Test_dfreqresp.test_from_zpk')
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_param_names_list', [])
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_dfreqresp.test_from_zpk.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.test_from_zpk', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_from_zpk', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_from_zpk(...)' code ##################

        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Assigning a Call to a Name (line 482):
        
        # Call to dlti(...): (line 482)
        # Processing the call arguments (line 482)
        
        # Obtaining an instance of the builtin type 'list' (line 482)
        list_294979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 482)
        
        
        # Obtaining an instance of the builtin type 'list' (line 482)
        list_294980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 482)
        # Adding element type (line 482)
        float_294981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 29), list_294980, float_294981)
        
        float_294982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 35), 'float')
        # Processing the call keyword arguments (line 482)
        kwargs_294983 = {}
        # Getting the type of 'dlti' (line 482)
        dlti_294978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 21), 'dlti', False)
        # Calling dlti(args, kwargs) (line 482)
        dlti_call_result_294984 = invoke(stypy.reporting.localization.Localization(__file__, 482, 21), dlti_294978, *[list_294979, list_294980, float_294982], **kwargs_294983)
        
        # Assigning a type to the variable 'system_ZPK' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'system_ZPK', dlti_call_result_294984)
        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Assigning a Call to a Name (line 483):
        
        # Call to dlti(...): (line 483)
        # Processing the call arguments (line 483)
        float_294986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 25), 'float')
        
        # Obtaining an instance of the builtin type 'list' (line 483)
        list_294987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 483)
        # Adding element type (line 483)
        int_294988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 30), list_294987, int_294988)
        # Adding element type (line 483)
        float_294989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 483, 30), list_294987, float_294989)
        
        # Processing the call keyword arguments (line 483)
        kwargs_294990 = {}
        # Getting the type of 'dlti' (line 483)
        dlti_294985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'dlti', False)
        # Calling dlti(args, kwargs) (line 483)
        dlti_call_result_294991 = invoke(stypy.reporting.localization.Localization(__file__, 483, 20), dlti_294985, *[float_294986, list_294987], **kwargs_294990)
        
        # Assigning a type to the variable 'system_TF' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'system_TF', dlti_call_result_294991)
        
        # Assigning a List to a Name (line 484):
        
        # Assigning a List to a Name (line 484):
        
        # Assigning a List to a Name (line 484):
        
        # Obtaining an instance of the builtin type 'list' (line 484)
        list_294992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 484)
        # Adding element type (line 484)
        float_294993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_294992, float_294993)
        # Adding element type (line 484)
        int_294994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_294992, int_294994)
        # Adding element type (line 484)
        int_294995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_294992, int_294995)
        # Adding element type (line 484)
        int_294996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 12), list_294992, int_294996)
        
        # Assigning a type to the variable 'w' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'w', list_294992)
        
        # Assigning a Call to a Tuple (line 485):
        
        # Assigning a Subscript to a Name (line 485):
        
        # Assigning a Subscript to a Name (line 485):
        
        # Obtaining the type of the subscript
        int_294997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 8), 'int')
        
        # Call to dfreqresp(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'system_ZPK' (line 485)
        system_ZPK_294999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 27), 'system_ZPK', False)
        # Processing the call keyword arguments (line 485)
        # Getting the type of 'w' (line 485)
        w_295000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 41), 'w', False)
        keyword_295001 = w_295000
        kwargs_295002 = {'w': keyword_295001}
        # Getting the type of 'dfreqresp' (line 485)
        dfreqresp_294998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 17), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 485)
        dfreqresp_call_result_295003 = invoke(stypy.reporting.localization.Localization(__file__, 485, 17), dfreqresp_294998, *[system_ZPK_294999], **kwargs_295002)
        
        # Obtaining the member '__getitem__' of a type (line 485)
        getitem___295004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), dfreqresp_call_result_295003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 485)
        subscript_call_result_295005 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), getitem___295004, int_294997)
        
        # Assigning a type to the variable 'tuple_var_assignment_292224' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'tuple_var_assignment_292224', subscript_call_result_295005)
        
        # Assigning a Subscript to a Name (line 485):
        
        # Assigning a Subscript to a Name (line 485):
        
        # Obtaining the type of the subscript
        int_295006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 8), 'int')
        
        # Call to dfreqresp(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'system_ZPK' (line 485)
        system_ZPK_295008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 27), 'system_ZPK', False)
        # Processing the call keyword arguments (line 485)
        # Getting the type of 'w' (line 485)
        w_295009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 41), 'w', False)
        keyword_295010 = w_295009
        kwargs_295011 = {'w': keyword_295010}
        # Getting the type of 'dfreqresp' (line 485)
        dfreqresp_295007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 17), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 485)
        dfreqresp_call_result_295012 = invoke(stypy.reporting.localization.Localization(__file__, 485, 17), dfreqresp_295007, *[system_ZPK_295008], **kwargs_295011)
        
        # Obtaining the member '__getitem__' of a type (line 485)
        getitem___295013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 8), dfreqresp_call_result_295012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 485)
        subscript_call_result_295014 = invoke(stypy.reporting.localization.Localization(__file__, 485, 8), getitem___295013, int_295006)
        
        # Assigning a type to the variable 'tuple_var_assignment_292225' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'tuple_var_assignment_292225', subscript_call_result_295014)
        
        # Assigning a Name to a Name (line 485):
        
        # Assigning a Name to a Name (line 485):
        # Getting the type of 'tuple_var_assignment_292224' (line 485)
        tuple_var_assignment_292224_295015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'tuple_var_assignment_292224')
        # Assigning a type to the variable 'w1' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'w1', tuple_var_assignment_292224_295015)
        
        # Assigning a Name to a Name (line 485):
        
        # Assigning a Name to a Name (line 485):
        # Getting the type of 'tuple_var_assignment_292225' (line 485)
        tuple_var_assignment_292225_295016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'tuple_var_assignment_292225')
        # Assigning a type to the variable 'H1' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'H1', tuple_var_assignment_292225_295016)
        
        # Assigning a Call to a Tuple (line 486):
        
        # Assigning a Subscript to a Name (line 486):
        
        # Assigning a Subscript to a Name (line 486):
        
        # Obtaining the type of the subscript
        int_295017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 8), 'int')
        
        # Call to dfreqresp(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'system_TF' (line 486)
        system_TF_295019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 27), 'system_TF', False)
        # Processing the call keyword arguments (line 486)
        # Getting the type of 'w' (line 486)
        w_295020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 40), 'w', False)
        keyword_295021 = w_295020
        kwargs_295022 = {'w': keyword_295021}
        # Getting the type of 'dfreqresp' (line 486)
        dfreqresp_295018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 486)
        dfreqresp_call_result_295023 = invoke(stypy.reporting.localization.Localization(__file__, 486, 17), dfreqresp_295018, *[system_TF_295019], **kwargs_295022)
        
        # Obtaining the member '__getitem__' of a type (line 486)
        getitem___295024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), dfreqresp_call_result_295023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 486)
        subscript_call_result_295025 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), getitem___295024, int_295017)
        
        # Assigning a type to the variable 'tuple_var_assignment_292226' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple_var_assignment_292226', subscript_call_result_295025)
        
        # Assigning a Subscript to a Name (line 486):
        
        # Assigning a Subscript to a Name (line 486):
        
        # Obtaining the type of the subscript
        int_295026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 486, 8), 'int')
        
        # Call to dfreqresp(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'system_TF' (line 486)
        system_TF_295028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 27), 'system_TF', False)
        # Processing the call keyword arguments (line 486)
        # Getting the type of 'w' (line 486)
        w_295029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 40), 'w', False)
        keyword_295030 = w_295029
        kwargs_295031 = {'w': keyword_295030}
        # Getting the type of 'dfreqresp' (line 486)
        dfreqresp_295027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 17), 'dfreqresp', False)
        # Calling dfreqresp(args, kwargs) (line 486)
        dfreqresp_call_result_295032 = invoke(stypy.reporting.localization.Localization(__file__, 486, 17), dfreqresp_295027, *[system_TF_295028], **kwargs_295031)
        
        # Obtaining the member '__getitem__' of a type (line 486)
        getitem___295033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 8), dfreqresp_call_result_295032, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 486)
        subscript_call_result_295034 = invoke(stypy.reporting.localization.Localization(__file__, 486, 8), getitem___295033, int_295026)
        
        # Assigning a type to the variable 'tuple_var_assignment_292227' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple_var_assignment_292227', subscript_call_result_295034)
        
        # Assigning a Name to a Name (line 486):
        
        # Assigning a Name to a Name (line 486):
        # Getting the type of 'tuple_var_assignment_292226' (line 486)
        tuple_var_assignment_292226_295035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple_var_assignment_292226')
        # Assigning a type to the variable 'w2' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'w2', tuple_var_assignment_292226_295035)
        
        # Assigning a Name to a Name (line 486):
        
        # Assigning a Name to a Name (line 486):
        # Getting the type of 'tuple_var_assignment_292227' (line 486)
        tuple_var_assignment_292227_295036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'tuple_var_assignment_292227')
        # Assigning a type to the variable 'H2' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'H2', tuple_var_assignment_292227_295036)
        
        # Call to assert_almost_equal(...): (line 487)
        # Processing the call arguments (line 487)
        # Getting the type of 'H1' (line 487)
        H1_295038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 28), 'H1', False)
        # Getting the type of 'H2' (line 487)
        H2_295039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 32), 'H2', False)
        # Processing the call keyword arguments (line 487)
        kwargs_295040 = {}
        # Getting the type of 'assert_almost_equal' (line 487)
        assert_almost_equal_295037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 487)
        assert_almost_equal_call_result_295041 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), assert_almost_equal_295037, *[H1_295038, H2_295039], **kwargs_295040)
        
        
        # ################# End of 'test_from_zpk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_from_zpk' in the type store
        # Getting the type of 'stypy_return_type' (line 480)
        stypy_return_type_295042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_from_zpk'
        return stypy_return_type_295042


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 399, 0, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_dfreqresp.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_dfreqresp' (line 399)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 0), 'Test_dfreqresp', Test_dfreqresp)
# Declaration of the 'Test_bode' class

class Test_bode(object, ):

    @norecursion
    def test_manual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_manual'
        module_type_store = module_type_store.open_function_context('test_manual', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_manual.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_manual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_manual.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_manual.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_manual')
        Test_bode.test_manual.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_manual.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_manual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_manual.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_manual.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_manual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_manual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_manual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_manual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_manual(...)' code ##################

        
        # Assigning a Num to a Name (line 495):
        
        # Assigning a Num to a Name (line 495):
        
        # Assigning a Num to a Name (line 495):
        float_295043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 13), 'float')
        # Assigning a type to the variable 'dt' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'dt', float_295043)
        
        # Assigning a Call to a Name (line 496):
        
        # Assigning a Call to a Name (line 496):
        
        # Assigning a Call to a Name (line 496):
        
        # Call to TransferFunction(...): (line 496)
        # Processing the call arguments (line 496)
        float_295045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 34), 'float')
        
        # Obtaining an instance of the builtin type 'list' (line 496)
        list_295046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 496)
        # Adding element type (line 496)
        int_295047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 39), list_295046, int_295047)
        # Adding element type (line 496)
        float_295048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 39), list_295046, float_295048)
        
        # Processing the call keyword arguments (line 496)
        # Getting the type of 'dt' (line 496)
        dt_295049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 53), 'dt', False)
        keyword_295050 = dt_295049
        kwargs_295051 = {'dt': keyword_295050}
        # Getting the type of 'TransferFunction' (line 496)
        TransferFunction_295044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 496)
        TransferFunction_call_result_295052 = invoke(stypy.reporting.localization.Localization(__file__, 496, 17), TransferFunction_295044, *[float_295045, list_295046], **kwargs_295051)
        
        # Assigning a type to the variable 'system' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'system', TransferFunction_call_result_295052)
        
        # Assigning a List to a Name (line 497):
        
        # Assigning a List to a Name (line 497):
        
        # Assigning a List to a Name (line 497):
        
        # Obtaining an instance of the builtin type 'list' (line 497)
        list_295053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 497)
        # Adding element type (line 497)
        float_295054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 12), list_295053, float_295054)
        # Adding element type (line 497)
        float_295055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 12), list_295053, float_295055)
        # Adding element type (line 497)
        int_295056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 12), list_295053, int_295056)
        # Adding element type (line 497)
        # Getting the type of 'np' (line 497)
        np_295057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 26), 'np')
        # Obtaining the member 'pi' of a type (line 497)
        pi_295058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 26), np_295057, 'pi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 12), list_295053, pi_295058)
        
        # Assigning a type to the variable 'w' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'w', list_295053)
        
        # Assigning a Call to a Tuple (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_295059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 8), 'int')
        
        # Call to dbode(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'system' (line 498)
        system_295061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'system', False)
        # Processing the call keyword arguments (line 498)
        # Getting the type of 'w' (line 498)
        w_295062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'w', False)
        keyword_295063 = w_295062
        kwargs_295064 = {'w': keyword_295063}
        # Getting the type of 'dbode' (line 498)
        dbode_295060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 498)
        dbode_call_result_295065 = invoke(stypy.reporting.localization.Localization(__file__, 498, 25), dbode_295060, *[system_295061], **kwargs_295064)
        
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___295066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), dbode_call_result_295065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_295067 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), getitem___295066, int_295059)
        
        # Assigning a type to the variable 'tuple_var_assignment_292228' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292228', subscript_call_result_295067)
        
        # Assigning a Subscript to a Name (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_295068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 8), 'int')
        
        # Call to dbode(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'system' (line 498)
        system_295070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'system', False)
        # Processing the call keyword arguments (line 498)
        # Getting the type of 'w' (line 498)
        w_295071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'w', False)
        keyword_295072 = w_295071
        kwargs_295073 = {'w': keyword_295072}
        # Getting the type of 'dbode' (line 498)
        dbode_295069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 498)
        dbode_call_result_295074 = invoke(stypy.reporting.localization.Localization(__file__, 498, 25), dbode_295069, *[system_295070], **kwargs_295073)
        
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___295075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), dbode_call_result_295074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_295076 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), getitem___295075, int_295068)
        
        # Assigning a type to the variable 'tuple_var_assignment_292229' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292229', subscript_call_result_295076)
        
        # Assigning a Subscript to a Name (line 498):
        
        # Assigning a Subscript to a Name (line 498):
        
        # Obtaining the type of the subscript
        int_295077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 8), 'int')
        
        # Call to dbode(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 'system' (line 498)
        system_295079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'system', False)
        # Processing the call keyword arguments (line 498)
        # Getting the type of 'w' (line 498)
        w_295080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 41), 'w', False)
        keyword_295081 = w_295080
        kwargs_295082 = {'w': keyword_295081}
        # Getting the type of 'dbode' (line 498)
        dbode_295078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 498)
        dbode_call_result_295083 = invoke(stypy.reporting.localization.Localization(__file__, 498, 25), dbode_295078, *[system_295079], **kwargs_295082)
        
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___295084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), dbode_call_result_295083, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_295085 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), getitem___295084, int_295077)
        
        # Assigning a type to the variable 'tuple_var_assignment_292230' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292230', subscript_call_result_295085)
        
        # Assigning a Name to a Name (line 498):
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_var_assignment_292228' (line 498)
        tuple_var_assignment_292228_295086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292228')
        # Assigning a type to the variable 'w2' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'w2', tuple_var_assignment_292228_295086)
        
        # Assigning a Name to a Name (line 498):
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_var_assignment_292229' (line 498)
        tuple_var_assignment_292229_295087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292229')
        # Assigning a type to the variable 'mag' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 12), 'mag', tuple_var_assignment_292229_295087)
        
        # Assigning a Name to a Name (line 498):
        
        # Assigning a Name to a Name (line 498):
        # Getting the type of 'tuple_var_assignment_292230' (line 498)
        tuple_var_assignment_292230_295088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'tuple_var_assignment_292230')
        # Assigning a type to the variable 'phase' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'phase', tuple_var_assignment_292230_295088)
        
        # Assigning a List to a Name (line 501):
        
        # Assigning a List to a Name (line 501):
        
        # Assigning a List to a Name (line 501):
        
        # Obtaining an instance of the builtin type 'list' (line 501)
        list_295089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 501)
        # Adding element type (line 501)
        float_295090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 23), list_295089, float_295090)
        # Adding element type (line 501)
        float_295091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 23), list_295089, float_295091)
        # Adding element type (line 501)
        float_295092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 23), list_295089, float_295092)
        # Adding element type (line 501)
        float_295093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 501, 23), list_295089, float_295093)
        
        # Assigning a type to the variable 'expected_mag' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'expected_mag', list_295089)
        
        # Call to assert_almost_equal(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 'mag' (line 502)
        mag_295095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'mag', False)
        # Getting the type of 'expected_mag' (line 502)
        expected_mag_295096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 33), 'expected_mag', False)
        # Processing the call keyword arguments (line 502)
        int_295097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 55), 'int')
        keyword_295098 = int_295097
        kwargs_295099 = {'decimal': keyword_295098}
        # Getting the type of 'assert_almost_equal' (line 502)
        assert_almost_equal_295094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 502)
        assert_almost_equal_call_result_295100 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), assert_almost_equal_295094, *[mag_295095, expected_mag_295096], **kwargs_295099)
        
        
        # Assigning a List to a Name (line 505):
        
        # Assigning a List to a Name (line 505):
        
        # Assigning a List to a Name (line 505):
        
        # Obtaining an instance of the builtin type 'list' (line 505)
        list_295101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 505)
        # Adding element type (line 505)
        float_295102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 25), list_295101, float_295102)
        # Adding element type (line 505)
        float_295103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 25), list_295101, float_295103)
        # Adding element type (line 505)
        float_295104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 25), list_295101, float_295104)
        # Adding element type (line 505)
        float_295105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 505, 25), list_295101, float_295105)
        
        # Assigning a type to the variable 'expected_phase' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'expected_phase', list_295101)
        
        # Call to assert_almost_equal(...): (line 506)
        # Processing the call arguments (line 506)
        # Getting the type of 'phase' (line 506)
        phase_295107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 28), 'phase', False)
        # Getting the type of 'expected_phase' (line 506)
        expected_phase_295108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 35), 'expected_phase', False)
        # Processing the call keyword arguments (line 506)
        int_295109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 59), 'int')
        keyword_295110 = int_295109
        kwargs_295111 = {'decimal': keyword_295110}
        # Getting the type of 'assert_almost_equal' (line 506)
        assert_almost_equal_295106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 506)
        assert_almost_equal_call_result_295112 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), assert_almost_equal_295106, *[phase_295107, expected_phase_295108], **kwargs_295111)
        
        
        # Call to assert_equal(...): (line 509)
        # Processing the call arguments (line 509)
        
        # Call to array(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'w' (line 509)
        w_295116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 30), 'w', False)
        # Processing the call keyword arguments (line 509)
        kwargs_295117 = {}
        # Getting the type of 'np' (line 509)
        np_295114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 509)
        array_295115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 21), np_295114, 'array')
        # Calling array(args, kwargs) (line 509)
        array_call_result_295118 = invoke(stypy.reporting.localization.Localization(__file__, 509, 21), array_295115, *[w_295116], **kwargs_295117)
        
        # Getting the type of 'dt' (line 509)
        dt_295119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 35), 'dt', False)
        # Applying the binary operator 'div' (line 509)
        result_div_295120 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 21), 'div', array_call_result_295118, dt_295119)
        
        # Getting the type of 'w2' (line 509)
        w2_295121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 39), 'w2', False)
        # Processing the call keyword arguments (line 509)
        kwargs_295122 = {}
        # Getting the type of 'assert_equal' (line 509)
        assert_equal_295113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 509)
        assert_equal_call_result_295123 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), assert_equal_295113, *[result_div_295120, w2_295121], **kwargs_295122)
        
        
        # ################# End of 'test_manual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_manual' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_295124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295124)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_manual'
        return stypy_return_type_295124


    @norecursion
    def test_auto(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_auto'
        module_type_store = module_type_store.open_function_context('test_auto', 511, 4, False)
        # Assigning a type to the variable 'self' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_auto.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_auto.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_auto.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_auto.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_auto')
        Test_bode.test_auto.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_auto.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_auto.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_auto.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_auto.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_auto.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_auto.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_auto', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_auto', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_auto(...)' code ##################

        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Assigning a Call to a Name (line 514):
        
        # Call to TransferFunction(...): (line 514)
        # Processing the call arguments (line 514)
        float_295126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 34), 'float')
        
        # Obtaining an instance of the builtin type 'list' (line 514)
        list_295127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 514)
        # Adding element type (line 514)
        int_295128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 39), list_295127, int_295128)
        # Adding element type (line 514)
        float_295129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 514, 39), list_295127, float_295129)
        
        # Processing the call keyword arguments (line 514)
        float_295130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 53), 'float')
        keyword_295131 = float_295130
        kwargs_295132 = {'dt': keyword_295131}
        # Getting the type of 'TransferFunction' (line 514)
        TransferFunction_295125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 514)
        TransferFunction_call_result_295133 = invoke(stypy.reporting.localization.Localization(__file__, 514, 17), TransferFunction_295125, *[float_295126, list_295127], **kwargs_295132)
        
        # Assigning a type to the variable 'system' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'system', TransferFunction_call_result_295133)
        
        # Assigning a Call to a Name (line 515):
        
        # Assigning a Call to a Name (line 515):
        
        # Assigning a Call to a Name (line 515):
        
        # Call to array(...): (line 515)
        # Processing the call arguments (line 515)
        
        # Obtaining an instance of the builtin type 'list' (line 515)
        list_295136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 515)
        # Adding element type (line 515)
        float_295137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), list_295136, float_295137)
        # Adding element type (line 515)
        float_295138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), list_295136, float_295138)
        # Adding element type (line 515)
        int_295139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), list_295136, int_295139)
        # Adding element type (line 515)
        # Getting the type of 'np' (line 515)
        np_295140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 35), 'np', False)
        # Obtaining the member 'pi' of a type (line 515)
        pi_295141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 35), np_295140, 'pi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 21), list_295136, pi_295141)
        
        # Processing the call keyword arguments (line 515)
        kwargs_295142 = {}
        # Getting the type of 'np' (line 515)
        np_295134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 515)
        array_295135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), np_295134, 'array')
        # Calling array(args, kwargs) (line 515)
        array_call_result_295143 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), array_295135, *[list_295136], **kwargs_295142)
        
        # Assigning a type to the variable 'w' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'w', array_call_result_295143)
        
        # Assigning a Call to a Tuple (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_295144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        
        # Call to dbode(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'system' (line 516)
        system_295146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 31), 'system', False)
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'w' (line 516)
        w_295147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 41), 'w', False)
        keyword_295148 = w_295147
        kwargs_295149 = {'w': keyword_295148}
        # Getting the type of 'dbode' (line 516)
        dbode_295145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 516)
        dbode_call_result_295150 = invoke(stypy.reporting.localization.Localization(__file__, 516, 25), dbode_295145, *[system_295146], **kwargs_295149)
        
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___295151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), dbode_call_result_295150, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_295152 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___295151, int_295144)
        
        # Assigning a type to the variable 'tuple_var_assignment_292231' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292231', subscript_call_result_295152)
        
        # Assigning a Subscript to a Name (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_295153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        
        # Call to dbode(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'system' (line 516)
        system_295155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 31), 'system', False)
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'w' (line 516)
        w_295156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 41), 'w', False)
        keyword_295157 = w_295156
        kwargs_295158 = {'w': keyword_295157}
        # Getting the type of 'dbode' (line 516)
        dbode_295154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 516)
        dbode_call_result_295159 = invoke(stypy.reporting.localization.Localization(__file__, 516, 25), dbode_295154, *[system_295155], **kwargs_295158)
        
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___295160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), dbode_call_result_295159, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_295161 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___295160, int_295153)
        
        # Assigning a type to the variable 'tuple_var_assignment_292232' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292232', subscript_call_result_295161)
        
        # Assigning a Subscript to a Name (line 516):
        
        # Assigning a Subscript to a Name (line 516):
        
        # Obtaining the type of the subscript
        int_295162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 8), 'int')
        
        # Call to dbode(...): (line 516)
        # Processing the call arguments (line 516)
        # Getting the type of 'system' (line 516)
        system_295164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 31), 'system', False)
        # Processing the call keyword arguments (line 516)
        # Getting the type of 'w' (line 516)
        w_295165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 41), 'w', False)
        keyword_295166 = w_295165
        kwargs_295167 = {'w': keyword_295166}
        # Getting the type of 'dbode' (line 516)
        dbode_295163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 25), 'dbode', False)
        # Calling dbode(args, kwargs) (line 516)
        dbode_call_result_295168 = invoke(stypy.reporting.localization.Localization(__file__, 516, 25), dbode_295163, *[system_295164], **kwargs_295167)
        
        # Obtaining the member '__getitem__' of a type (line 516)
        getitem___295169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), dbode_call_result_295168, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 516)
        subscript_call_result_295170 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), getitem___295169, int_295162)
        
        # Assigning a type to the variable 'tuple_var_assignment_292233' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292233', subscript_call_result_295170)
        
        # Assigning a Name to a Name (line 516):
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_292231' (line 516)
        tuple_var_assignment_292231_295171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292231')
        # Assigning a type to the variable 'w2' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'w2', tuple_var_assignment_292231_295171)
        
        # Assigning a Name to a Name (line 516):
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_292232' (line 516)
        tuple_var_assignment_292232_295172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292232')
        # Assigning a type to the variable 'mag' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'mag', tuple_var_assignment_292232_295172)
        
        # Assigning a Name to a Name (line 516):
        
        # Assigning a Name to a Name (line 516):
        # Getting the type of 'tuple_var_assignment_292233' (line 516)
        tuple_var_assignment_292233_295173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'tuple_var_assignment_292233')
        # Assigning a type to the variable 'phase' (line 516)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 17), 'phase', tuple_var_assignment_292233_295173)
        
        # Assigning a Call to a Name (line 517):
        
        # Assigning a Call to a Name (line 517):
        
        # Assigning a Call to a Name (line 517):
        
        # Call to exp(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 'w' (line 517)
        w_295176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 20), 'w', False)
        complex_295177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 24), 'complex')
        # Applying the binary operator '*' (line 517)
        result_mul_295178 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 20), '*', w_295176, complex_295177)
        
        # Processing the call keyword arguments (line 517)
        kwargs_295179 = {}
        # Getting the type of 'np' (line 517)
        np_295174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 13), 'np', False)
        # Obtaining the member 'exp' of a type (line 517)
        exp_295175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 13), np_295174, 'exp')
        # Calling exp(args, kwargs) (line 517)
        exp_call_result_295180 = invoke(stypy.reporting.localization.Localization(__file__, 517, 13), exp_295175, *[result_mul_295178], **kwargs_295179)
        
        # Assigning a type to the variable 'jw' (line 517)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'jw', exp_call_result_295180)
        
        # Assigning a BinOp to a Name (line 518):
        
        # Assigning a BinOp to a Name (line 518):
        
        # Assigning a BinOp to a Name (line 518):
        
        # Call to polyval(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'system' (line 518)
        system_295183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 23), 'system', False)
        # Obtaining the member 'num' of a type (line 518)
        num_295184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 23), system_295183, 'num')
        # Getting the type of 'jw' (line 518)
        jw_295185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 35), 'jw', False)
        # Processing the call keyword arguments (line 518)
        kwargs_295186 = {}
        # Getting the type of 'np' (line 518)
        np_295181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'np', False)
        # Obtaining the member 'polyval' of a type (line 518)
        polyval_295182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), np_295181, 'polyval')
        # Calling polyval(args, kwargs) (line 518)
        polyval_call_result_295187 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), polyval_295182, *[num_295184, jw_295185], **kwargs_295186)
        
        
        # Call to polyval(...): (line 518)
        # Processing the call arguments (line 518)
        # Getting the type of 'system' (line 518)
        system_295190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 52), 'system', False)
        # Obtaining the member 'den' of a type (line 518)
        den_295191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 52), system_295190, 'den')
        # Getting the type of 'jw' (line 518)
        jw_295192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 64), 'jw', False)
        # Processing the call keyword arguments (line 518)
        kwargs_295193 = {}
        # Getting the type of 'np' (line 518)
        np_295188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 41), 'np', False)
        # Obtaining the member 'polyval' of a type (line 518)
        polyval_295189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 41), np_295188, 'polyval')
        # Calling polyval(args, kwargs) (line 518)
        polyval_call_result_295194 = invoke(stypy.reporting.localization.Localization(__file__, 518, 41), polyval_295189, *[den_295191, jw_295192], **kwargs_295193)
        
        # Applying the binary operator 'div' (line 518)
        result_div_295195 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 12), 'div', polyval_call_result_295187, polyval_call_result_295194)
        
        # Assigning a type to the variable 'y' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'y', result_div_295195)
        
        # Assigning a BinOp to a Name (line 521):
        
        # Assigning a BinOp to a Name (line 521):
        
        # Assigning a BinOp to a Name (line 521):
        float_295196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 23), 'float')
        
        # Call to log10(...): (line 521)
        # Processing the call arguments (line 521)
        
        # Call to abs(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'y' (line 521)
        y_295200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 43), 'y', False)
        # Processing the call keyword arguments (line 521)
        kwargs_295201 = {}
        # Getting the type of 'abs' (line 521)
        abs_295199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 39), 'abs', False)
        # Calling abs(args, kwargs) (line 521)
        abs_call_result_295202 = invoke(stypy.reporting.localization.Localization(__file__, 521, 39), abs_295199, *[y_295200], **kwargs_295201)
        
        # Processing the call keyword arguments (line 521)
        kwargs_295203 = {}
        # Getting the type of 'np' (line 521)
        np_295197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 30), 'np', False)
        # Obtaining the member 'log10' of a type (line 521)
        log10_295198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 30), np_295197, 'log10')
        # Calling log10(args, kwargs) (line 521)
        log10_call_result_295204 = invoke(stypy.reporting.localization.Localization(__file__, 521, 30), log10_295198, *[abs_call_result_295202], **kwargs_295203)
        
        # Applying the binary operator '*' (line 521)
        result_mul_295205 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 23), '*', float_295196, log10_call_result_295204)
        
        # Assigning a type to the variable 'expected_mag' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'expected_mag', result_mul_295205)
        
        # Call to assert_almost_equal(...): (line 522)
        # Processing the call arguments (line 522)
        # Getting the type of 'mag' (line 522)
        mag_295207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 28), 'mag', False)
        # Getting the type of 'expected_mag' (line 522)
        expected_mag_295208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 33), 'expected_mag', False)
        # Processing the call keyword arguments (line 522)
        kwargs_295209 = {}
        # Getting the type of 'assert_almost_equal' (line 522)
        assert_almost_equal_295206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 522)
        assert_almost_equal_call_result_295210 = invoke(stypy.reporting.localization.Localization(__file__, 522, 8), assert_almost_equal_295206, *[mag_295207, expected_mag_295208], **kwargs_295209)
        
        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Call to rad2deg(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Call to angle(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'y' (line 525)
        y_295215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 45), 'y', False)
        # Processing the call keyword arguments (line 525)
        kwargs_295216 = {}
        # Getting the type of 'np' (line 525)
        np_295213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 36), 'np', False)
        # Obtaining the member 'angle' of a type (line 525)
        angle_295214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 36), np_295213, 'angle')
        # Calling angle(args, kwargs) (line 525)
        angle_call_result_295217 = invoke(stypy.reporting.localization.Localization(__file__, 525, 36), angle_295214, *[y_295215], **kwargs_295216)
        
        # Processing the call keyword arguments (line 525)
        kwargs_295218 = {}
        # Getting the type of 'np' (line 525)
        np_295211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'np', False)
        # Obtaining the member 'rad2deg' of a type (line 525)
        rad2deg_295212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 25), np_295211, 'rad2deg')
        # Calling rad2deg(args, kwargs) (line 525)
        rad2deg_call_result_295219 = invoke(stypy.reporting.localization.Localization(__file__, 525, 25), rad2deg_295212, *[angle_call_result_295217], **kwargs_295218)
        
        # Assigning a type to the variable 'expected_phase' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'expected_phase', rad2deg_call_result_295219)
        
        # Call to assert_almost_equal(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'phase' (line 526)
        phase_295221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 28), 'phase', False)
        # Getting the type of 'expected_phase' (line 526)
        expected_phase_295222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 35), 'expected_phase', False)
        # Processing the call keyword arguments (line 526)
        kwargs_295223 = {}
        # Getting the type of 'assert_almost_equal' (line 526)
        assert_almost_equal_295220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 526)
        assert_almost_equal_call_result_295224 = invoke(stypy.reporting.localization.Localization(__file__, 526, 8), assert_almost_equal_295220, *[phase_295221, expected_phase_295222], **kwargs_295223)
        
        
        # ################# End of 'test_auto(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_auto' in the type store
        # Getting the type of 'stypy_return_type' (line 511)
        stypy_return_type_295225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_auto'
        return stypy_return_type_295225


    @norecursion
    def test_range(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_range'
        module_type_store = module_type_store.open_function_context('test_range', 528, 4, False)
        # Assigning a type to the variable 'self' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_range.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_range.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_range.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_range.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_range')
        Test_bode.test_range.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_range.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_range.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_range.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_range.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_range.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_range.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_range', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_range', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_range(...)' code ##################

        
        # Assigning a Num to a Name (line 531):
        
        # Assigning a Num to a Name (line 531):
        
        # Assigning a Num to a Name (line 531):
        float_295226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 13), 'float')
        # Assigning a type to the variable 'dt' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'dt', float_295226)
        
        # Assigning a Call to a Name (line 532):
        
        # Assigning a Call to a Name (line 532):
        
        # Assigning a Call to a Name (line 532):
        
        # Call to TransferFunction(...): (line 532)
        # Processing the call arguments (line 532)
        float_295228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 34), 'float')
        
        # Obtaining an instance of the builtin type 'list' (line 532)
        list_295229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 532)
        # Adding element type (line 532)
        int_295230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 39), list_295229, int_295230)
        # Adding element type (line 532)
        float_295231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 39), list_295229, float_295231)
        
        # Processing the call keyword arguments (line 532)
        float_295232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 53), 'float')
        keyword_295233 = float_295232
        kwargs_295234 = {'dt': keyword_295233}
        # Getting the type of 'TransferFunction' (line 532)
        TransferFunction_295227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 532)
        TransferFunction_call_result_295235 = invoke(stypy.reporting.localization.Localization(__file__, 532, 17), TransferFunction_295227, *[float_295228, list_295229], **kwargs_295234)
        
        # Assigning a type to the variable 'system' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'system', TransferFunction_call_result_295235)
        
        # Assigning a Num to a Name (line 533):
        
        # Assigning a Num to a Name (line 533):
        
        # Assigning a Num to a Name (line 533):
        int_295236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 12), 'int')
        # Assigning a type to the variable 'n' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'n', int_295236)
        
        # Assigning a BinOp to a Name (line 535):
        
        # Assigning a BinOp to a Name (line 535):
        
        # Assigning a BinOp to a Name (line 535):
        
        # Call to linspace(...): (line 535)
        # Processing the call arguments (line 535)
        int_295239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 33), 'int')
        # Getting the type of 'np' (line 535)
        np_295240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 36), 'np', False)
        # Obtaining the member 'pi' of a type (line 535)
        pi_295241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 36), np_295240, 'pi')
        # Getting the type of 'n' (line 535)
        n_295242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 43), 'n', False)
        # Processing the call keyword arguments (line 535)
        # Getting the type of 'False' (line 535)
        False_295243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 55), 'False', False)
        keyword_295244 = False_295243
        kwargs_295245 = {'endpoint': keyword_295244}
        # Getting the type of 'np' (line 535)
        np_295237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 21), 'np', False)
        # Obtaining the member 'linspace' of a type (line 535)
        linspace_295238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 21), np_295237, 'linspace')
        # Calling linspace(args, kwargs) (line 535)
        linspace_call_result_295246 = invoke(stypy.reporting.localization.Localization(__file__, 535, 21), linspace_295238, *[int_295239, pi_295241, n_295242], **kwargs_295245)
        
        # Getting the type of 'dt' (line 535)
        dt_295247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 64), 'dt')
        # Applying the binary operator 'div' (line 535)
        result_div_295248 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 21), 'div', linspace_call_result_295246, dt_295247)
        
        # Assigning a type to the variable 'expected_w' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'expected_w', result_div_295248)
        
        # Assigning a Call to a Tuple (line 536):
        
        # Assigning a Subscript to a Name (line 536):
        
        # Assigning a Subscript to a Name (line 536):
        
        # Obtaining the type of the subscript
        int_295249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 8), 'int')
        
        # Call to dbode(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'system' (line 536)
        system_295251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 30), 'system', False)
        # Processing the call keyword arguments (line 536)
        # Getting the type of 'n' (line 536)
        n_295252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 40), 'n', False)
        keyword_295253 = n_295252
        kwargs_295254 = {'n': keyword_295253}
        # Getting the type of 'dbode' (line 536)
        dbode_295250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'dbode', False)
        # Calling dbode(args, kwargs) (line 536)
        dbode_call_result_295255 = invoke(stypy.reporting.localization.Localization(__file__, 536, 24), dbode_295250, *[system_295251], **kwargs_295254)
        
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___295256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), dbode_call_result_295255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_295257 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), getitem___295256, int_295249)
        
        # Assigning a type to the variable 'tuple_var_assignment_292234' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292234', subscript_call_result_295257)
        
        # Assigning a Subscript to a Name (line 536):
        
        # Assigning a Subscript to a Name (line 536):
        
        # Obtaining the type of the subscript
        int_295258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 8), 'int')
        
        # Call to dbode(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'system' (line 536)
        system_295260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 30), 'system', False)
        # Processing the call keyword arguments (line 536)
        # Getting the type of 'n' (line 536)
        n_295261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 40), 'n', False)
        keyword_295262 = n_295261
        kwargs_295263 = {'n': keyword_295262}
        # Getting the type of 'dbode' (line 536)
        dbode_295259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'dbode', False)
        # Calling dbode(args, kwargs) (line 536)
        dbode_call_result_295264 = invoke(stypy.reporting.localization.Localization(__file__, 536, 24), dbode_295259, *[system_295260], **kwargs_295263)
        
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___295265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), dbode_call_result_295264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_295266 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), getitem___295265, int_295258)
        
        # Assigning a type to the variable 'tuple_var_assignment_292235' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292235', subscript_call_result_295266)
        
        # Assigning a Subscript to a Name (line 536):
        
        # Assigning a Subscript to a Name (line 536):
        
        # Obtaining the type of the subscript
        int_295267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 8), 'int')
        
        # Call to dbode(...): (line 536)
        # Processing the call arguments (line 536)
        # Getting the type of 'system' (line 536)
        system_295269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 30), 'system', False)
        # Processing the call keyword arguments (line 536)
        # Getting the type of 'n' (line 536)
        n_295270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 40), 'n', False)
        keyword_295271 = n_295270
        kwargs_295272 = {'n': keyword_295271}
        # Getting the type of 'dbode' (line 536)
        dbode_295268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'dbode', False)
        # Calling dbode(args, kwargs) (line 536)
        dbode_call_result_295273 = invoke(stypy.reporting.localization.Localization(__file__, 536, 24), dbode_295268, *[system_295269], **kwargs_295272)
        
        # Obtaining the member '__getitem__' of a type (line 536)
        getitem___295274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 8), dbode_call_result_295273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 536)
        subscript_call_result_295275 = invoke(stypy.reporting.localization.Localization(__file__, 536, 8), getitem___295274, int_295267)
        
        # Assigning a type to the variable 'tuple_var_assignment_292236' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292236', subscript_call_result_295275)
        
        # Assigning a Name to a Name (line 536):
        
        # Assigning a Name to a Name (line 536):
        # Getting the type of 'tuple_var_assignment_292234' (line 536)
        tuple_var_assignment_292234_295276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292234')
        # Assigning a type to the variable 'w' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'w', tuple_var_assignment_292234_295276)
        
        # Assigning a Name to a Name (line 536):
        
        # Assigning a Name to a Name (line 536):
        # Getting the type of 'tuple_var_assignment_292235' (line 536)
        tuple_var_assignment_292235_295277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292235')
        # Assigning a type to the variable 'mag' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 11), 'mag', tuple_var_assignment_292235_295277)
        
        # Assigning a Name to a Name (line 536):
        
        # Assigning a Name to a Name (line 536):
        # Getting the type of 'tuple_var_assignment_292236' (line 536)
        tuple_var_assignment_292236_295278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'tuple_var_assignment_292236')
        # Assigning a type to the variable 'phase' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'phase', tuple_var_assignment_292236_295278)
        
        # Call to assert_almost_equal(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'w' (line 537)
        w_295280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'w', False)
        # Getting the type of 'expected_w' (line 537)
        expected_w_295281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 31), 'expected_w', False)
        # Processing the call keyword arguments (line 537)
        kwargs_295282 = {}
        # Getting the type of 'assert_almost_equal' (line 537)
        assert_almost_equal_295279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 537)
        assert_almost_equal_call_result_295283 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), assert_almost_equal_295279, *[w_295280, expected_w_295281], **kwargs_295282)
        
        
        # ################# End of 'test_range(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_range' in the type store
        # Getting the type of 'stypy_return_type' (line 528)
        stypy_return_type_295284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_range'
        return stypy_return_type_295284


    @norecursion
    def test_pole_one(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pole_one'
        module_type_store = module_type_store.open_function_context('test_pole_one', 539, 4, False)
        # Assigning a type to the variable 'self' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_pole_one')
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_pole_one.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_pole_one', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pole_one', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pole_one(...)' code ##################

        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Assigning a Call to a Name (line 542):
        
        # Call to TransferFunction(...): (line 542)
        # Processing the call arguments (line 542)
        
        # Obtaining an instance of the builtin type 'list' (line 542)
        list_295286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 542)
        # Adding element type (line 542)
        int_295287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 34), list_295286, int_295287)
        
        
        # Obtaining an instance of the builtin type 'list' (line 542)
        list_295288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 542)
        # Adding element type (line 542)
        int_295289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 39), list_295288, int_295289)
        # Adding element type (line 542)
        int_295290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 542, 39), list_295288, int_295290)
        
        # Processing the call keyword arguments (line 542)
        float_295291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 51), 'float')
        keyword_295292 = float_295291
        kwargs_295293 = {'dt': keyword_295292}
        # Getting the type of 'TransferFunction' (line 542)
        TransferFunction_295285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 542)
        TransferFunction_call_result_295294 = invoke(stypy.reporting.localization.Localization(__file__, 542, 17), TransferFunction_295285, *[list_295286, list_295288], **kwargs_295293)
        
        # Assigning a type to the variable 'system' (line 542)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'system', TransferFunction_call_result_295294)
        
        # Call to suppress_warnings(...): (line 544)
        # Processing the call keyword arguments (line 544)
        kwargs_295296 = {}
        # Getting the type of 'suppress_warnings' (line 544)
        suppress_warnings_295295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 544)
        suppress_warnings_call_result_295297 = invoke(stypy.reporting.localization.Localization(__file__, 544, 13), suppress_warnings_295295, *[], **kwargs_295296)
        
        with_295298 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 544, 13), suppress_warnings_call_result_295297, 'with parameter', '__enter__', '__exit__')

        if with_295298:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 544)
            enter___295299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 13), suppress_warnings_call_result_295297, '__enter__')
            with_enter_295300 = invoke(stypy.reporting.localization.Localization(__file__, 544, 13), enter___295299)
            # Assigning a type to the variable 'sup' (line 544)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 13), 'sup', with_enter_295300)
            
            # Call to filter(...): (line 545)
            # Processing the call arguments (line 545)
            # Getting the type of 'RuntimeWarning' (line 545)
            RuntimeWarning_295303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 545)
            str_295304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 47), 'str', 'divide by zero')
            keyword_295305 = str_295304
            kwargs_295306 = {'message': keyword_295305}
            # Getting the type of 'sup' (line 545)
            sup_295301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 545)
            filter_295302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 12), sup_295301, 'filter')
            # Calling filter(args, kwargs) (line 545)
            filter_call_result_295307 = invoke(stypy.reporting.localization.Localization(__file__, 545, 12), filter_295302, *[RuntimeWarning_295303], **kwargs_295306)
            
            
            # Call to filter(...): (line 546)
            # Processing the call arguments (line 546)
            # Getting the type of 'RuntimeWarning' (line 546)
            RuntimeWarning_295310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 23), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 546)
            str_295311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 47), 'str', 'invalid value encountered')
            keyword_295312 = str_295311
            kwargs_295313 = {'message': keyword_295312}
            # Getting the type of 'sup' (line 546)
            sup_295308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 546)
            filter_295309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 12), sup_295308, 'filter')
            # Calling filter(args, kwargs) (line 546)
            filter_call_result_295314 = invoke(stypy.reporting.localization.Localization(__file__, 546, 12), filter_295309, *[RuntimeWarning_295310], **kwargs_295313)
            
            
            # Assigning a Call to a Tuple (line 547):
            
            # Assigning a Subscript to a Name (line 547):
            
            # Assigning a Subscript to a Name (line 547):
            
            # Obtaining the type of the subscript
            int_295315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 12), 'int')
            
            # Call to dbode(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'system' (line 547)
            system_295317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 34), 'system', False)
            # Processing the call keyword arguments (line 547)
            int_295318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 44), 'int')
            keyword_295319 = int_295318
            kwargs_295320 = {'n': keyword_295319}
            # Getting the type of 'dbode' (line 547)
            dbode_295316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'dbode', False)
            # Calling dbode(args, kwargs) (line 547)
            dbode_call_result_295321 = invoke(stypy.reporting.localization.Localization(__file__, 547, 28), dbode_295316, *[system_295317], **kwargs_295320)
            
            # Obtaining the member '__getitem__' of a type (line 547)
            getitem___295322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), dbode_call_result_295321, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 547)
            subscript_call_result_295323 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), getitem___295322, int_295315)
            
            # Assigning a type to the variable 'tuple_var_assignment_292237' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292237', subscript_call_result_295323)
            
            # Assigning a Subscript to a Name (line 547):
            
            # Assigning a Subscript to a Name (line 547):
            
            # Obtaining the type of the subscript
            int_295324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 12), 'int')
            
            # Call to dbode(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'system' (line 547)
            system_295326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 34), 'system', False)
            # Processing the call keyword arguments (line 547)
            int_295327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 44), 'int')
            keyword_295328 = int_295327
            kwargs_295329 = {'n': keyword_295328}
            # Getting the type of 'dbode' (line 547)
            dbode_295325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'dbode', False)
            # Calling dbode(args, kwargs) (line 547)
            dbode_call_result_295330 = invoke(stypy.reporting.localization.Localization(__file__, 547, 28), dbode_295325, *[system_295326], **kwargs_295329)
            
            # Obtaining the member '__getitem__' of a type (line 547)
            getitem___295331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), dbode_call_result_295330, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 547)
            subscript_call_result_295332 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), getitem___295331, int_295324)
            
            # Assigning a type to the variable 'tuple_var_assignment_292238' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292238', subscript_call_result_295332)
            
            # Assigning a Subscript to a Name (line 547):
            
            # Assigning a Subscript to a Name (line 547):
            
            # Obtaining the type of the subscript
            int_295333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 12), 'int')
            
            # Call to dbode(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'system' (line 547)
            system_295335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 34), 'system', False)
            # Processing the call keyword arguments (line 547)
            int_295336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 44), 'int')
            keyword_295337 = int_295336
            kwargs_295338 = {'n': keyword_295337}
            # Getting the type of 'dbode' (line 547)
            dbode_295334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 28), 'dbode', False)
            # Calling dbode(args, kwargs) (line 547)
            dbode_call_result_295339 = invoke(stypy.reporting.localization.Localization(__file__, 547, 28), dbode_295334, *[system_295335], **kwargs_295338)
            
            # Obtaining the member '__getitem__' of a type (line 547)
            getitem___295340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 12), dbode_call_result_295339, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 547)
            subscript_call_result_295341 = invoke(stypy.reporting.localization.Localization(__file__, 547, 12), getitem___295340, int_295333)
            
            # Assigning a type to the variable 'tuple_var_assignment_292239' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292239', subscript_call_result_295341)
            
            # Assigning a Name to a Name (line 547):
            
            # Assigning a Name to a Name (line 547):
            # Getting the type of 'tuple_var_assignment_292237' (line 547)
            tuple_var_assignment_292237_295342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292237')
            # Assigning a type to the variable 'w' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'w', tuple_var_assignment_292237_295342)
            
            # Assigning a Name to a Name (line 547):
            
            # Assigning a Name to a Name (line 547):
            # Getting the type of 'tuple_var_assignment_292238' (line 547)
            tuple_var_assignment_292238_295343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292238')
            # Assigning a type to the variable 'mag' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'mag', tuple_var_assignment_292238_295343)
            
            # Assigning a Name to a Name (line 547):
            
            # Assigning a Name to a Name (line 547):
            # Getting the type of 'tuple_var_assignment_292239' (line 547)
            tuple_var_assignment_292239_295344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'tuple_var_assignment_292239')
            # Assigning a type to the variable 'phase' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 20), 'phase', tuple_var_assignment_292239_295344)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 544)
            exit___295345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 13), suppress_warnings_call_result_295297, '__exit__')
            with_exit_295346 = invoke(stypy.reporting.localization.Localization(__file__, 544, 13), exit___295345, None, None, None)

        
        # Call to assert_equal(...): (line 548)
        # Processing the call arguments (line 548)
        
        # Obtaining the type of the subscript
        int_295348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 23), 'int')
        # Getting the type of 'w' (line 548)
        w_295349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 21), 'w', False)
        # Obtaining the member '__getitem__' of a type (line 548)
        getitem___295350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 21), w_295349, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 548)
        subscript_call_result_295351 = invoke(stypy.reporting.localization.Localization(__file__, 548, 21), getitem___295350, int_295348)
        
        float_295352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 27), 'float')
        # Processing the call keyword arguments (line 548)
        kwargs_295353 = {}
        # Getting the type of 'assert_equal' (line 548)
        assert_equal_295347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 548)
        assert_equal_call_result_295354 = invoke(stypy.reporting.localization.Localization(__file__, 548, 8), assert_equal_295347, *[subscript_call_result_295351, float_295352], **kwargs_295353)
        
        
        # ################# End of 'test_pole_one(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pole_one' in the type store
        # Getting the type of 'stypy_return_type' (line 539)
        stypy_return_type_295355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295355)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pole_one'
        return stypy_return_type_295355


    @norecursion
    def test_imaginary(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_imaginary'
        module_type_store = module_type_store.open_function_context('test_imaginary', 550, 4, False)
        # Assigning a type to the variable 'self' (line 551)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_imaginary')
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_imaginary.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_imaginary', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_imaginary', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_imaginary(...)' code ##################

        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to TransferFunction(...): (line 553)
        # Processing the call arguments (line 553)
        
        # Obtaining an instance of the builtin type 'list' (line 553)
        list_295357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 553)
        # Adding element type (line 553)
        int_295358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 34), list_295357, int_295358)
        
        
        # Obtaining an instance of the builtin type 'list' (line 553)
        list_295359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 553)
        # Adding element type (line 553)
        int_295360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 39), list_295359, int_295360)
        # Adding element type (line 553)
        int_295361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 39), list_295359, int_295361)
        # Adding element type (line 553)
        int_295362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 39), list_295359, int_295362)
        
        # Processing the call keyword arguments (line 553)
        float_295363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 55), 'float')
        keyword_295364 = float_295363
        kwargs_295365 = {'dt': keyword_295364}
        # Getting the type of 'TransferFunction' (line 553)
        TransferFunction_295356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 17), 'TransferFunction', False)
        # Calling TransferFunction(args, kwargs) (line 553)
        TransferFunction_call_result_295366 = invoke(stypy.reporting.localization.Localization(__file__, 553, 17), TransferFunction_295356, *[list_295357, list_295359], **kwargs_295365)
        
        # Assigning a type to the variable 'system' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'system', TransferFunction_call_result_295366)
        
        # Call to dbode(...): (line 554)
        # Processing the call arguments (line 554)
        # Getting the type of 'system' (line 554)
        system_295368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 14), 'system', False)
        # Processing the call keyword arguments (line 554)
        int_295369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 24), 'int')
        keyword_295370 = int_295369
        kwargs_295371 = {'n': keyword_295370}
        # Getting the type of 'dbode' (line 554)
        dbode_295367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'dbode', False)
        # Calling dbode(args, kwargs) (line 554)
        dbode_call_result_295372 = invoke(stypy.reporting.localization.Localization(__file__, 554, 8), dbode_295367, *[system_295368], **kwargs_295371)
        
        
        # ################# End of 'test_imaginary(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_imaginary' in the type store
        # Getting the type of 'stypy_return_type' (line 550)
        stypy_return_type_295373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_imaginary'
        return stypy_return_type_295373


    @norecursion
    def test_error(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_error'
        module_type_store = module_type_store.open_function_context('test_error', 556, 4, False)
        # Assigning a type to the variable 'self' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Test_bode.test_error.__dict__.__setitem__('stypy_localization', localization)
        Test_bode.test_error.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Test_bode.test_error.__dict__.__setitem__('stypy_type_store', module_type_store)
        Test_bode.test_error.__dict__.__setitem__('stypy_function_name', 'Test_bode.test_error')
        Test_bode.test_error.__dict__.__setitem__('stypy_param_names_list', [])
        Test_bode.test_error.__dict__.__setitem__('stypy_varargs_param_name', None)
        Test_bode.test_error.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Test_bode.test_error.__dict__.__setitem__('stypy_call_defaults', defaults)
        Test_bode.test_error.__dict__.__setitem__('stypy_call_varargs', varargs)
        Test_bode.test_error.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Test_bode.test_error.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.test_error', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_error', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_error(...)' code ##################

        
        # Assigning a Call to a Name (line 558):
        
        # Assigning a Call to a Name (line 558):
        
        # Assigning a Call to a Name (line 558):
        
        # Call to lti(...): (line 558)
        # Processing the call arguments (line 558)
        
        # Obtaining an instance of the builtin type 'list' (line 558)
        list_295375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 558)
        # Adding element type (line 558)
        int_295376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 21), list_295375, int_295376)
        
        
        # Obtaining an instance of the builtin type 'list' (line 558)
        list_295377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 558)
        # Adding element type (line 558)
        int_295378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 26), list_295377, int_295378)
        # Adding element type (line 558)
        int_295379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 558, 26), list_295377, int_295379)
        
        # Processing the call keyword arguments (line 558)
        kwargs_295380 = {}
        # Getting the type of 'lti' (line 558)
        lti_295374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 17), 'lti', False)
        # Calling lti(args, kwargs) (line 558)
        lti_call_result_295381 = invoke(stypy.reporting.localization.Localization(__file__, 558, 17), lti_295374, *[list_295375, list_295377], **kwargs_295380)
        
        # Assigning a type to the variable 'system' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'system', lti_call_result_295381)
        
        # Call to assert_raises(...): (line 559)
        # Processing the call arguments (line 559)
        # Getting the type of 'AttributeError' (line 559)
        AttributeError_295383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 22), 'AttributeError', False)
        # Getting the type of 'dbode' (line 559)
        dbode_295384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 38), 'dbode', False)
        # Getting the type of 'system' (line 559)
        system_295385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 45), 'system', False)
        # Processing the call keyword arguments (line 559)
        kwargs_295386 = {}
        # Getting the type of 'assert_raises' (line 559)
        assert_raises_295382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 559)
        assert_raises_call_result_295387 = invoke(stypy.reporting.localization.Localization(__file__, 559, 8), assert_raises_295382, *[AttributeError_295383, dbode_295384, system_295385], **kwargs_295386)
        
        
        # ################# End of 'test_error(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_error' in the type store
        # Getting the type of 'stypy_return_type' (line 556)
        stypy_return_type_295388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295388)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_error'
        return stypy_return_type_295388


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 490, 0, False)
        # Assigning a type to the variable 'self' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test_bode.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Test_bode' (line 490)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 0), 'Test_bode', Test_bode)
# Declaration of the 'TestTransferFunctionZConversion' class

class TestTransferFunctionZConversion(object, ):
    str_295389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 4), 'str', "Test private conversions between 'z' and 'z**-1' polynomials.")

    @norecursion
    def test_full(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_full'
        module_type_store = module_type_store.open_function_context('test_full', 565, 4, False)
        # Assigning a type to the variable 'self' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_function_name', 'TestTransferFunctionZConversion.test_full')
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunctionZConversion.test_full.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunctionZConversion.test_full', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_full', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_full(...)' code ##################

        
        # Assigning a List to a Name (line 567):
        
        # Assigning a List to a Name (line 567):
        
        # Assigning a List to a Name (line 567):
        
        # Obtaining an instance of the builtin type 'list' (line 567)
        list_295390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 567)
        # Adding element type (line 567)
        int_295391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 14), list_295390, int_295391)
        # Adding element type (line 567)
        int_295392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 14), list_295390, int_295392)
        # Adding element type (line 567)
        int_295393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 14), list_295390, int_295393)
        
        # Assigning a type to the variable 'num' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'num', list_295390)
        
        # Assigning a List to a Name (line 568):
        
        # Assigning a List to a Name (line 568):
        
        # Assigning a List to a Name (line 568):
        
        # Obtaining an instance of the builtin type 'list' (line 568)
        list_295394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 568)
        # Adding element type (line 568)
        int_295395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 14), list_295394, int_295395)
        # Adding element type (line 568)
        int_295396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 14), list_295394, int_295396)
        # Adding element type (line 568)
        int_295397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 14), list_295394, int_295397)
        
        # Assigning a type to the variable 'den' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'den', list_295394)
        
        # Assigning a Call to a Tuple (line 569):
        
        # Assigning a Subscript to a Name (line 569):
        
        # Assigning a Subscript to a Name (line 569):
        
        # Obtaining the type of the subscript
        int_295398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'num' (line 569)
        num_295401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 49), 'num', False)
        # Getting the type of 'den' (line 569)
        den_295402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 54), 'den', False)
        # Processing the call keyword arguments (line 569)
        kwargs_295403 = {}
        # Getting the type of 'TransferFunction' (line 569)
        TransferFunction_295399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 569)
        _z_to_zinv_295400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), TransferFunction_295399, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 569)
        _z_to_zinv_call_result_295404 = invoke(stypy.reporting.localization.Localization(__file__, 569, 21), _z_to_zinv_295400, *[num_295401, den_295402], **kwargs_295403)
        
        # Obtaining the member '__getitem__' of a type (line 569)
        getitem___295405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), _z_to_zinv_call_result_295404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 569)
        subscript_call_result_295406 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), getitem___295405, int_295398)
        
        # Assigning a type to the variable 'tuple_var_assignment_292240' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'tuple_var_assignment_292240', subscript_call_result_295406)
        
        # Assigning a Subscript to a Name (line 569):
        
        # Assigning a Subscript to a Name (line 569):
        
        # Obtaining the type of the subscript
        int_295407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 'num' (line 569)
        num_295410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 49), 'num', False)
        # Getting the type of 'den' (line 569)
        den_295411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 54), 'den', False)
        # Processing the call keyword arguments (line 569)
        kwargs_295412 = {}
        # Getting the type of 'TransferFunction' (line 569)
        TransferFunction_295408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 569)
        _z_to_zinv_295409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 21), TransferFunction_295408, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 569)
        _z_to_zinv_call_result_295413 = invoke(stypy.reporting.localization.Localization(__file__, 569, 21), _z_to_zinv_295409, *[num_295410, den_295411], **kwargs_295412)
        
        # Obtaining the member '__getitem__' of a type (line 569)
        getitem___295414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), _z_to_zinv_call_result_295413, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 569)
        subscript_call_result_295415 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), getitem___295414, int_295407)
        
        # Assigning a type to the variable 'tuple_var_assignment_292241' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'tuple_var_assignment_292241', subscript_call_result_295415)
        
        # Assigning a Name to a Name (line 569):
        
        # Assigning a Name to a Name (line 569):
        # Getting the type of 'tuple_var_assignment_292240' (line 569)
        tuple_var_assignment_292240_295416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'tuple_var_assignment_292240')
        # Assigning a type to the variable 'num2' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'num2', tuple_var_assignment_292240_295416)
        
        # Assigning a Name to a Name (line 569):
        
        # Assigning a Name to a Name (line 569):
        # Getting the type of 'tuple_var_assignment_292241' (line 569)
        tuple_var_assignment_292241_295417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'tuple_var_assignment_292241')
        # Assigning a type to the variable 'den2' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 14), 'den2', tuple_var_assignment_292241_295417)
        
        # Call to assert_equal(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'num' (line 570)
        num_295419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 21), 'num', False)
        # Getting the type of 'num2' (line 570)
        num2_295420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 26), 'num2', False)
        # Processing the call keyword arguments (line 570)
        kwargs_295421 = {}
        # Getting the type of 'assert_equal' (line 570)
        assert_equal_295418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 570)
        assert_equal_call_result_295422 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), assert_equal_295418, *[num_295419, num2_295420], **kwargs_295421)
        
        
        # Call to assert_equal(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 'den' (line 571)
        den_295424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'den', False)
        # Getting the type of 'den2' (line 571)
        den2_295425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 26), 'den2', False)
        # Processing the call keyword arguments (line 571)
        kwargs_295426 = {}
        # Getting the type of 'assert_equal' (line 571)
        assert_equal_295423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 571)
        assert_equal_call_result_295427 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), assert_equal_295423, *[den_295424, den2_295425], **kwargs_295426)
        
        
        # Assigning a Call to a Tuple (line 573):
        
        # Assigning a Subscript to a Name (line 573):
        
        # Assigning a Subscript to a Name (line 573):
        
        # Obtaining the type of the subscript
        int_295428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'num' (line 573)
        num_295431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 49), 'num', False)
        # Getting the type of 'den' (line 573)
        den_295432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 54), 'den', False)
        # Processing the call keyword arguments (line 573)
        kwargs_295433 = {}
        # Getting the type of 'TransferFunction' (line 573)
        TransferFunction_295429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 573)
        _zinv_to_z_295430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 21), TransferFunction_295429, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 573)
        _zinv_to_z_call_result_295434 = invoke(stypy.reporting.localization.Localization(__file__, 573, 21), _zinv_to_z_295430, *[num_295431, den_295432], **kwargs_295433)
        
        # Obtaining the member '__getitem__' of a type (line 573)
        getitem___295435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), _zinv_to_z_call_result_295434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 573)
        subscript_call_result_295436 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), getitem___295435, int_295428)
        
        # Assigning a type to the variable 'tuple_var_assignment_292242' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'tuple_var_assignment_292242', subscript_call_result_295436)
        
        # Assigning a Subscript to a Name (line 573):
        
        # Assigning a Subscript to a Name (line 573):
        
        # Obtaining the type of the subscript
        int_295437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 573)
        # Processing the call arguments (line 573)
        # Getting the type of 'num' (line 573)
        num_295440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 49), 'num', False)
        # Getting the type of 'den' (line 573)
        den_295441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 54), 'den', False)
        # Processing the call keyword arguments (line 573)
        kwargs_295442 = {}
        # Getting the type of 'TransferFunction' (line 573)
        TransferFunction_295438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 573)
        _zinv_to_z_295439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 21), TransferFunction_295438, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 573)
        _zinv_to_z_call_result_295443 = invoke(stypy.reporting.localization.Localization(__file__, 573, 21), _zinv_to_z_295439, *[num_295440, den_295441], **kwargs_295442)
        
        # Obtaining the member '__getitem__' of a type (line 573)
        getitem___295444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 8), _zinv_to_z_call_result_295443, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 573)
        subscript_call_result_295445 = invoke(stypy.reporting.localization.Localization(__file__, 573, 8), getitem___295444, int_295437)
        
        # Assigning a type to the variable 'tuple_var_assignment_292243' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'tuple_var_assignment_292243', subscript_call_result_295445)
        
        # Assigning a Name to a Name (line 573):
        
        # Assigning a Name to a Name (line 573):
        # Getting the type of 'tuple_var_assignment_292242' (line 573)
        tuple_var_assignment_292242_295446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'tuple_var_assignment_292242')
        # Assigning a type to the variable 'num2' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'num2', tuple_var_assignment_292242_295446)
        
        # Assigning a Name to a Name (line 573):
        
        # Assigning a Name to a Name (line 573):
        # Getting the type of 'tuple_var_assignment_292243' (line 573)
        tuple_var_assignment_292243_295447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'tuple_var_assignment_292243')
        # Assigning a type to the variable 'den2' (line 573)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 14), 'den2', tuple_var_assignment_292243_295447)
        
        # Call to assert_equal(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'num' (line 574)
        num_295449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 21), 'num', False)
        # Getting the type of 'num2' (line 574)
        num2_295450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 26), 'num2', False)
        # Processing the call keyword arguments (line 574)
        kwargs_295451 = {}
        # Getting the type of 'assert_equal' (line 574)
        assert_equal_295448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 574)
        assert_equal_call_result_295452 = invoke(stypy.reporting.localization.Localization(__file__, 574, 8), assert_equal_295448, *[num_295449, num2_295450], **kwargs_295451)
        
        
        # Call to assert_equal(...): (line 575)
        # Processing the call arguments (line 575)
        # Getting the type of 'den' (line 575)
        den_295454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 21), 'den', False)
        # Getting the type of 'den2' (line 575)
        den2_295455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 26), 'den2', False)
        # Processing the call keyword arguments (line 575)
        kwargs_295456 = {}
        # Getting the type of 'assert_equal' (line 575)
        assert_equal_295453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 575)
        assert_equal_call_result_295457 = invoke(stypy.reporting.localization.Localization(__file__, 575, 8), assert_equal_295453, *[den_295454, den2_295455], **kwargs_295456)
        
        
        # ################# End of 'test_full(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_full' in the type store
        # Getting the type of 'stypy_return_type' (line 565)
        stypy_return_type_295458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_full'
        return stypy_return_type_295458


    @norecursion
    def test_numerator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_numerator'
        module_type_store = module_type_store.open_function_context('test_numerator', 577, 4, False)
        # Assigning a type to the variable 'self' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_function_name', 'TestTransferFunctionZConversion.test_numerator')
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunctionZConversion.test_numerator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunctionZConversion.test_numerator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_numerator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_numerator(...)' code ##################

        
        # Assigning a List to a Name (line 579):
        
        # Assigning a List to a Name (line 579):
        
        # Assigning a List to a Name (line 579):
        
        # Obtaining an instance of the builtin type 'list' (line 579)
        list_295459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 579)
        # Adding element type (line 579)
        int_295460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 14), list_295459, int_295460)
        # Adding element type (line 579)
        int_295461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 14), list_295459, int_295461)
        
        # Assigning a type to the variable 'num' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'num', list_295459)
        
        # Assigning a List to a Name (line 580):
        
        # Assigning a List to a Name (line 580):
        
        # Assigning a List to a Name (line 580):
        
        # Obtaining an instance of the builtin type 'list' (line 580)
        list_295462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 580)
        # Adding element type (line 580)
        int_295463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 14), list_295462, int_295463)
        # Adding element type (line 580)
        int_295464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 14), list_295462, int_295464)
        # Adding element type (line 580)
        int_295465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 14), list_295462, int_295465)
        
        # Assigning a type to the variable 'den' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'den', list_295462)
        
        # Assigning a Call to a Tuple (line 581):
        
        # Assigning a Subscript to a Name (line 581):
        
        # Assigning a Subscript to a Name (line 581):
        
        # Obtaining the type of the subscript
        int_295466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'num' (line 581)
        num_295469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 49), 'num', False)
        # Getting the type of 'den' (line 581)
        den_295470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'den', False)
        # Processing the call keyword arguments (line 581)
        kwargs_295471 = {}
        # Getting the type of 'TransferFunction' (line 581)
        TransferFunction_295467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 581)
        _z_to_zinv_295468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 21), TransferFunction_295467, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 581)
        _z_to_zinv_call_result_295472 = invoke(stypy.reporting.localization.Localization(__file__, 581, 21), _z_to_zinv_295468, *[num_295469, den_295470], **kwargs_295471)
        
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___295473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), _z_to_zinv_call_result_295472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_295474 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___295473, int_295466)
        
        # Assigning a type to the variable 'tuple_var_assignment_292244' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_292244', subscript_call_result_295474)
        
        # Assigning a Subscript to a Name (line 581):
        
        # Assigning a Subscript to a Name (line 581):
        
        # Obtaining the type of the subscript
        int_295475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 581)
        # Processing the call arguments (line 581)
        # Getting the type of 'num' (line 581)
        num_295478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 49), 'num', False)
        # Getting the type of 'den' (line 581)
        den_295479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'den', False)
        # Processing the call keyword arguments (line 581)
        kwargs_295480 = {}
        # Getting the type of 'TransferFunction' (line 581)
        TransferFunction_295476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 581)
        _z_to_zinv_295477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 21), TransferFunction_295476, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 581)
        _z_to_zinv_call_result_295481 = invoke(stypy.reporting.localization.Localization(__file__, 581, 21), _z_to_zinv_295477, *[num_295478, den_295479], **kwargs_295480)
        
        # Obtaining the member '__getitem__' of a type (line 581)
        getitem___295482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), _z_to_zinv_call_result_295481, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 581)
        subscript_call_result_295483 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___295482, int_295475)
        
        # Assigning a type to the variable 'tuple_var_assignment_292245' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_292245', subscript_call_result_295483)
        
        # Assigning a Name to a Name (line 581):
        
        # Assigning a Name to a Name (line 581):
        # Getting the type of 'tuple_var_assignment_292244' (line 581)
        tuple_var_assignment_292244_295484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_292244')
        # Assigning a type to the variable 'num2' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'num2', tuple_var_assignment_292244_295484)
        
        # Assigning a Name to a Name (line 581):
        
        # Assigning a Name to a Name (line 581):
        # Getting the type of 'tuple_var_assignment_292245' (line 581)
        tuple_var_assignment_292245_295485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_292245')
        # Assigning a type to the variable 'den2' (line 581)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'den2', tuple_var_assignment_292245_295485)
        
        # Call to assert_equal(...): (line 582)
        # Processing the call arguments (line 582)
        
        # Obtaining an instance of the builtin type 'list' (line 582)
        list_295487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 582)
        # Adding element type (line 582)
        int_295488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 21), list_295487, int_295488)
        # Adding element type (line 582)
        int_295489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 21), list_295487, int_295489)
        # Adding element type (line 582)
        int_295490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 21), list_295487, int_295490)
        
        # Getting the type of 'num2' (line 582)
        num2_295491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 32), 'num2', False)
        # Processing the call keyword arguments (line 582)
        kwargs_295492 = {}
        # Getting the type of 'assert_equal' (line 582)
        assert_equal_295486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 582)
        assert_equal_call_result_295493 = invoke(stypy.reporting.localization.Localization(__file__, 582, 8), assert_equal_295486, *[list_295487, num2_295491], **kwargs_295492)
        
        
        # Call to assert_equal(...): (line 583)
        # Processing the call arguments (line 583)
        # Getting the type of 'den' (line 583)
        den_295495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 21), 'den', False)
        # Getting the type of 'den2' (line 583)
        den2_295496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 26), 'den2', False)
        # Processing the call keyword arguments (line 583)
        kwargs_295497 = {}
        # Getting the type of 'assert_equal' (line 583)
        assert_equal_295494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 583)
        assert_equal_call_result_295498 = invoke(stypy.reporting.localization.Localization(__file__, 583, 8), assert_equal_295494, *[den_295495, den2_295496], **kwargs_295497)
        
        
        # Assigning a Call to a Tuple (line 585):
        
        # Assigning a Subscript to a Name (line 585):
        
        # Assigning a Subscript to a Name (line 585):
        
        # Obtaining the type of the subscript
        int_295499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'num' (line 585)
        num_295502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 49), 'num', False)
        # Getting the type of 'den' (line 585)
        den_295503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 54), 'den', False)
        # Processing the call keyword arguments (line 585)
        kwargs_295504 = {}
        # Getting the type of 'TransferFunction' (line 585)
        TransferFunction_295500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 585)
        _zinv_to_z_295501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 21), TransferFunction_295500, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 585)
        _zinv_to_z_call_result_295505 = invoke(stypy.reporting.localization.Localization(__file__, 585, 21), _zinv_to_z_295501, *[num_295502, den_295503], **kwargs_295504)
        
        # Obtaining the member '__getitem__' of a type (line 585)
        getitem___295506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), _zinv_to_z_call_result_295505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 585)
        subscript_call_result_295507 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), getitem___295506, int_295499)
        
        # Assigning a type to the variable 'tuple_var_assignment_292246' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_292246', subscript_call_result_295507)
        
        # Assigning a Subscript to a Name (line 585):
        
        # Assigning a Subscript to a Name (line 585):
        
        # Obtaining the type of the subscript
        int_295508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 'num' (line 585)
        num_295511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 49), 'num', False)
        # Getting the type of 'den' (line 585)
        den_295512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 54), 'den', False)
        # Processing the call keyword arguments (line 585)
        kwargs_295513 = {}
        # Getting the type of 'TransferFunction' (line 585)
        TransferFunction_295509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 585)
        _zinv_to_z_295510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 21), TransferFunction_295509, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 585)
        _zinv_to_z_call_result_295514 = invoke(stypy.reporting.localization.Localization(__file__, 585, 21), _zinv_to_z_295510, *[num_295511, den_295512], **kwargs_295513)
        
        # Obtaining the member '__getitem__' of a type (line 585)
        getitem___295515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), _zinv_to_z_call_result_295514, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 585)
        subscript_call_result_295516 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), getitem___295515, int_295508)
        
        # Assigning a type to the variable 'tuple_var_assignment_292247' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_292247', subscript_call_result_295516)
        
        # Assigning a Name to a Name (line 585):
        
        # Assigning a Name to a Name (line 585):
        # Getting the type of 'tuple_var_assignment_292246' (line 585)
        tuple_var_assignment_292246_295517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_292246')
        # Assigning a type to the variable 'num2' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'num2', tuple_var_assignment_292246_295517)
        
        # Assigning a Name to a Name (line 585):
        
        # Assigning a Name to a Name (line 585):
        # Getting the type of 'tuple_var_assignment_292247' (line 585)
        tuple_var_assignment_292247_295518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_292247')
        # Assigning a type to the variable 'den2' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 14), 'den2', tuple_var_assignment_292247_295518)
        
        # Call to assert_equal(...): (line 586)
        # Processing the call arguments (line 586)
        
        # Obtaining an instance of the builtin type 'list' (line 586)
        list_295520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 586)
        # Adding element type (line 586)
        int_295521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 21), list_295520, int_295521)
        # Adding element type (line 586)
        int_295522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 21), list_295520, int_295522)
        # Adding element type (line 586)
        int_295523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 21), list_295520, int_295523)
        
        # Getting the type of 'num2' (line 586)
        num2_295524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 32), 'num2', False)
        # Processing the call keyword arguments (line 586)
        kwargs_295525 = {}
        # Getting the type of 'assert_equal' (line 586)
        assert_equal_295519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 586)
        assert_equal_call_result_295526 = invoke(stypy.reporting.localization.Localization(__file__, 586, 8), assert_equal_295519, *[list_295520, num2_295524], **kwargs_295525)
        
        
        # Call to assert_equal(...): (line 587)
        # Processing the call arguments (line 587)
        # Getting the type of 'den' (line 587)
        den_295528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 21), 'den', False)
        # Getting the type of 'den2' (line 587)
        den2_295529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'den2', False)
        # Processing the call keyword arguments (line 587)
        kwargs_295530 = {}
        # Getting the type of 'assert_equal' (line 587)
        assert_equal_295527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 587)
        assert_equal_call_result_295531 = invoke(stypy.reporting.localization.Localization(__file__, 587, 8), assert_equal_295527, *[den_295528, den2_295529], **kwargs_295530)
        
        
        # ################# End of 'test_numerator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_numerator' in the type store
        # Getting the type of 'stypy_return_type' (line 577)
        stypy_return_type_295532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_numerator'
        return stypy_return_type_295532


    @norecursion
    def test_denominator(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_denominator'
        module_type_store = module_type_store.open_function_context('test_denominator', 589, 4, False)
        # Assigning a type to the variable 'self' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_localization', localization)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_function_name', 'TestTransferFunctionZConversion.test_denominator')
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_param_names_list', [])
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTransferFunctionZConversion.test_denominator.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunctionZConversion.test_denominator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_denominator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_denominator(...)' code ##################

        
        # Assigning a List to a Name (line 591):
        
        # Assigning a List to a Name (line 591):
        
        # Assigning a List to a Name (line 591):
        
        # Obtaining an instance of the builtin type 'list' (line 591)
        list_295533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 591)
        # Adding element type (line 591)
        int_295534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 14), list_295533, int_295534)
        # Adding element type (line 591)
        int_295535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 14), list_295533, int_295535)
        # Adding element type (line 591)
        int_295536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 14), list_295533, int_295536)
        
        # Assigning a type to the variable 'num' (line 591)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'num', list_295533)
        
        # Assigning a List to a Name (line 592):
        
        # Assigning a List to a Name (line 592):
        
        # Assigning a List to a Name (line 592):
        
        # Obtaining an instance of the builtin type 'list' (line 592)
        list_295537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 592)
        # Adding element type (line 592)
        int_295538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 14), list_295537, int_295538)
        # Adding element type (line 592)
        int_295539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 592, 14), list_295537, int_295539)
        
        # Assigning a type to the variable 'den' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'den', list_295537)
        
        # Assigning a Call to a Tuple (line 593):
        
        # Assigning a Subscript to a Name (line 593):
        
        # Assigning a Subscript to a Name (line 593):
        
        # Obtaining the type of the subscript
        int_295540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'num' (line 593)
        num_295543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 49), 'num', False)
        # Getting the type of 'den' (line 593)
        den_295544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 54), 'den', False)
        # Processing the call keyword arguments (line 593)
        kwargs_295545 = {}
        # Getting the type of 'TransferFunction' (line 593)
        TransferFunction_295541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 593)
        _z_to_zinv_295542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 21), TransferFunction_295541, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 593)
        _z_to_zinv_call_result_295546 = invoke(stypy.reporting.localization.Localization(__file__, 593, 21), _z_to_zinv_295542, *[num_295543, den_295544], **kwargs_295545)
        
        # Obtaining the member '__getitem__' of a type (line 593)
        getitem___295547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _z_to_zinv_call_result_295546, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 593)
        subscript_call_result_295548 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___295547, int_295540)
        
        # Assigning a type to the variable 'tuple_var_assignment_292248' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_292248', subscript_call_result_295548)
        
        # Assigning a Subscript to a Name (line 593):
        
        # Assigning a Subscript to a Name (line 593):
        
        # Obtaining the type of the subscript
        int_295549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 8), 'int')
        
        # Call to _z_to_zinv(...): (line 593)
        # Processing the call arguments (line 593)
        # Getting the type of 'num' (line 593)
        num_295552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 49), 'num', False)
        # Getting the type of 'den' (line 593)
        den_295553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 54), 'den', False)
        # Processing the call keyword arguments (line 593)
        kwargs_295554 = {}
        # Getting the type of 'TransferFunction' (line 593)
        TransferFunction_295550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 21), 'TransferFunction', False)
        # Obtaining the member '_z_to_zinv' of a type (line 593)
        _z_to_zinv_295551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 21), TransferFunction_295550, '_z_to_zinv')
        # Calling _z_to_zinv(args, kwargs) (line 593)
        _z_to_zinv_call_result_295555 = invoke(stypy.reporting.localization.Localization(__file__, 593, 21), _z_to_zinv_295551, *[num_295552, den_295553], **kwargs_295554)
        
        # Obtaining the member '__getitem__' of a type (line 593)
        getitem___295556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 8), _z_to_zinv_call_result_295555, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 593)
        subscript_call_result_295557 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), getitem___295556, int_295549)
        
        # Assigning a type to the variable 'tuple_var_assignment_292249' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_292249', subscript_call_result_295557)
        
        # Assigning a Name to a Name (line 593):
        
        # Assigning a Name to a Name (line 593):
        # Getting the type of 'tuple_var_assignment_292248' (line 593)
        tuple_var_assignment_292248_295558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_292248')
        # Assigning a type to the variable 'num2' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'num2', tuple_var_assignment_292248_295558)
        
        # Assigning a Name to a Name (line 593):
        
        # Assigning a Name to a Name (line 593):
        # Getting the type of 'tuple_var_assignment_292249' (line 593)
        tuple_var_assignment_292249_295559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'tuple_var_assignment_292249')
        # Assigning a type to the variable 'den2' (line 593)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 14), 'den2', tuple_var_assignment_292249_295559)
        
        # Call to assert_equal(...): (line 594)
        # Processing the call arguments (line 594)
        # Getting the type of 'num' (line 594)
        num_295561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 21), 'num', False)
        # Getting the type of 'num2' (line 594)
        num2_295562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 26), 'num2', False)
        # Processing the call keyword arguments (line 594)
        kwargs_295563 = {}
        # Getting the type of 'assert_equal' (line 594)
        assert_equal_295560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 594)
        assert_equal_call_result_295564 = invoke(stypy.reporting.localization.Localization(__file__, 594, 8), assert_equal_295560, *[num_295561, num2_295562], **kwargs_295563)
        
        
        # Call to assert_equal(...): (line 595)
        # Processing the call arguments (line 595)
        
        # Obtaining an instance of the builtin type 'list' (line 595)
        list_295566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 595)
        # Adding element type (line 595)
        int_295567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 21), list_295566, int_295567)
        # Adding element type (line 595)
        int_295568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 21), list_295566, int_295568)
        # Adding element type (line 595)
        int_295569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 21), list_295566, int_295569)
        
        # Getting the type of 'den2' (line 595)
        den2_295570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 32), 'den2', False)
        # Processing the call keyword arguments (line 595)
        kwargs_295571 = {}
        # Getting the type of 'assert_equal' (line 595)
        assert_equal_295565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 595)
        assert_equal_call_result_295572 = invoke(stypy.reporting.localization.Localization(__file__, 595, 8), assert_equal_295565, *[list_295566, den2_295570], **kwargs_295571)
        
        
        # Assigning a Call to a Tuple (line 597):
        
        # Assigning a Subscript to a Name (line 597):
        
        # Assigning a Subscript to a Name (line 597):
        
        # Obtaining the type of the subscript
        int_295573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'num' (line 597)
        num_295576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 49), 'num', False)
        # Getting the type of 'den' (line 597)
        den_295577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 54), 'den', False)
        # Processing the call keyword arguments (line 597)
        kwargs_295578 = {}
        # Getting the type of 'TransferFunction' (line 597)
        TransferFunction_295574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 597)
        _zinv_to_z_295575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 21), TransferFunction_295574, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 597)
        _zinv_to_z_call_result_295579 = invoke(stypy.reporting.localization.Localization(__file__, 597, 21), _zinv_to_z_295575, *[num_295576, den_295577], **kwargs_295578)
        
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___295580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), _zinv_to_z_call_result_295579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_295581 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), getitem___295580, int_295573)
        
        # Assigning a type to the variable 'tuple_var_assignment_292250' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_292250', subscript_call_result_295581)
        
        # Assigning a Subscript to a Name (line 597):
        
        # Assigning a Subscript to a Name (line 597):
        
        # Obtaining the type of the subscript
        int_295582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 8), 'int')
        
        # Call to _zinv_to_z(...): (line 597)
        # Processing the call arguments (line 597)
        # Getting the type of 'num' (line 597)
        num_295585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 49), 'num', False)
        # Getting the type of 'den' (line 597)
        den_295586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 54), 'den', False)
        # Processing the call keyword arguments (line 597)
        kwargs_295587 = {}
        # Getting the type of 'TransferFunction' (line 597)
        TransferFunction_295583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 21), 'TransferFunction', False)
        # Obtaining the member '_zinv_to_z' of a type (line 597)
        _zinv_to_z_295584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 21), TransferFunction_295583, '_zinv_to_z')
        # Calling _zinv_to_z(args, kwargs) (line 597)
        _zinv_to_z_call_result_295588 = invoke(stypy.reporting.localization.Localization(__file__, 597, 21), _zinv_to_z_295584, *[num_295585, den_295586], **kwargs_295587)
        
        # Obtaining the member '__getitem__' of a type (line 597)
        getitem___295589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 597, 8), _zinv_to_z_call_result_295588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 597)
        subscript_call_result_295590 = invoke(stypy.reporting.localization.Localization(__file__, 597, 8), getitem___295589, int_295582)
        
        # Assigning a type to the variable 'tuple_var_assignment_292251' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_292251', subscript_call_result_295590)
        
        # Assigning a Name to a Name (line 597):
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'tuple_var_assignment_292250' (line 597)
        tuple_var_assignment_292250_295591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_292250')
        # Assigning a type to the variable 'num2' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'num2', tuple_var_assignment_292250_295591)
        
        # Assigning a Name to a Name (line 597):
        
        # Assigning a Name to a Name (line 597):
        # Getting the type of 'tuple_var_assignment_292251' (line 597)
        tuple_var_assignment_292251_295592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'tuple_var_assignment_292251')
        # Assigning a type to the variable 'den2' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 14), 'den2', tuple_var_assignment_292251_295592)
        
        # Call to assert_equal(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 'num' (line 598)
        num_295594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 21), 'num', False)
        # Getting the type of 'num2' (line 598)
        num2_295595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 26), 'num2', False)
        # Processing the call keyword arguments (line 598)
        kwargs_295596 = {}
        # Getting the type of 'assert_equal' (line 598)
        assert_equal_295593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 598)
        assert_equal_call_result_295597 = invoke(stypy.reporting.localization.Localization(__file__, 598, 8), assert_equal_295593, *[num_295594, num2_295595], **kwargs_295596)
        
        
        # Call to assert_equal(...): (line 599)
        # Processing the call arguments (line 599)
        
        # Obtaining an instance of the builtin type 'list' (line 599)
        list_295599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 599)
        # Adding element type (line 599)
        int_295600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 21), list_295599, int_295600)
        # Adding element type (line 599)
        int_295601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 21), list_295599, int_295601)
        # Adding element type (line 599)
        int_295602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 21), list_295599, int_295602)
        
        # Getting the type of 'den2' (line 599)
        den2_295603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 32), 'den2', False)
        # Processing the call keyword arguments (line 599)
        kwargs_295604 = {}
        # Getting the type of 'assert_equal' (line 599)
        assert_equal_295598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 599)
        assert_equal_call_result_295605 = invoke(stypy.reporting.localization.Localization(__file__, 599, 8), assert_equal_295598, *[list_295599, den2_295603], **kwargs_295604)
        
        
        # ################# End of 'test_denominator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_denominator' in the type store
        # Getting the type of 'stypy_return_type' (line 589)
        stypy_return_type_295606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_295606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_denominator'
        return stypy_return_type_295606


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 562, 0, False)
        # Assigning a type to the variable 'self' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTransferFunctionZConversion.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTransferFunctionZConversion' (line 562)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 0), 'TestTransferFunctionZConversion', TestTransferFunctionZConversion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
