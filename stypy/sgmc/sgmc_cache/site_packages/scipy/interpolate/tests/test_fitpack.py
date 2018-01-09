
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: import numpy as np
6: from numpy.testing import (assert_equal, assert_allclose, assert_,
7:                            assert_almost_equal, assert_array_almost_equal)
8: from pytest import raises as assert_raises
9: 
10: from numpy import array, asarray, pi, sin, cos, arange, dot, ravel, sqrt, round
11: from scipy import interpolate
12: from scipy.interpolate.fitpack import (splrep, splev, bisplrep, bisplev,
13:      sproot, splprep, splint, spalde, splder, splantider, insert, dblint)
14: from scipy.interpolate.dfitpack import regrid_smth
15: 
16: 
17: def data_file(basename):
18:     return os.path.join(os.path.abspath(os.path.dirname(__file__)),
19:                         'data', basename)
20: 
21: 
22: def norm2(x):
23:     return sqrt(dot(x.T,x))
24: 
25: 
26: def f1(x,d=0):
27:     if d is None:
28:         return "sin"
29:     if x is None:
30:         return "sin(x)"
31:     if d % 4 == 0:
32:         return sin(x)
33:     if d % 4 == 1:
34:         return cos(x)
35:     if d % 4 == 2:
36:         return -sin(x)
37:     if d % 4 == 3:
38:         return -cos(x)
39: 
40: 
41: def f2(x,y=0,dx=0,dy=0):
42:     if x is None:
43:         return "sin(x+y)"
44:     d = dx+dy
45:     if d % 4 == 0:
46:         return sin(x+y)
47:     if d % 4 == 1:
48:         return cos(x+y)
49:     if d % 4 == 2:
50:         return -sin(x+y)
51:     if d % 4 == 3:
52:         return -cos(x+y)
53: 
54: 
55: def makepairs(x, y):
56:     '''Helper function to create an array of pairs of x and y.'''
57:     # Or itertools.product (>= python 2.6)
58:     xy = array([[a, b] for a in asarray(x) for b in asarray(y)])
59:     return xy.T
60: 
61: 
62: def put(*a):
63:     '''Produce some output if file run directly'''
64:     import sys
65:     if hasattr(sys.modules['__main__'], '__put_prints'):
66:         sys.stderr.write("".join(map(str, a)) + "\n")
67: 
68: 
69: class TestSmokeTests(object):
70:     '''
71:     Smoke tests (with a few asserts) for fitpack routines -- mostly
72:     check that they are runnable
73:     '''
74: 
75:     def check_1(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,at=0,xb=None,xe=None):
76:         if xb is None:
77:             xb = a
78:         if xe is None:
79:             xe = b
80:         x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
81:         x1 = a+(b-a)*arange(1,N,dtype=float)/float(N-1)  # middle points of the nodes
82:         v,v1 = f(x),f(x1)
83:         nk = []
84: 
85:         def err_est(k, d):
86:             # Assume f has all derivatives < 1
87:             h = 1.0/float(N)
88:             tol = 5 * h**(.75*(k-d))
89:             if s > 0:
90:                 tol += 1e5*s
91:             return tol
92: 
93:         for k in range(1,6):
94:             tck = splrep(x,v,s=s,per=per,k=k,xe=xe)
95:             if at:
96:                 t = tck[0][k:-k]
97:             else:
98:                 t = x1
99:             nd = []
100:             for d in range(k+1):
101:                 tol = err_est(k, d)
102:                 err = norm2(f(t,d)-splev(t,tck,d)) / norm2(f(t,d))
103:                 assert_(err < tol, (k, d, err, tol))
104:                 nd.append((err, tol))
105:             nk.append(nd)
106:         put("\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]" % (f(None),
107:                                                         repr(round(xb,3)),repr(round(xe,3)),
108:                                                           repr(round(a,3)),repr(round(b,3))))
109:         if at:
110:             str = "at knots"
111:         else:
112:             str = "at the middle of nodes"
113:         put(" per=%d s=%s Evaluation %s" % (per,repr(s),str))
114:         put(" k :  |f-s|^2  |f'-s'| |f''-.. |f'''-. |f''''- |f'''''")
115:         k = 1
116:         for l in nk:
117:             put(' %d : ' % k)
118:             for r in l:
119:                 put(' %.1e  %.1e' % r)
120:             put('\n')
121:             k = k+1
122: 
123:     def check_2(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
124:               ia=0,ib=2*pi,dx=0.2*pi):
125:         if xb is None:
126:             xb = a
127:         if xe is None:
128:             xe = b
129:         x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
130:         v = f(x)
131: 
132:         def err_est(k, d):
133:             # Assume f has all derivatives < 1
134:             h = 1.0/float(N)
135:             tol = 5 * h**(.75*(k-d))
136:             if s > 0:
137:                 tol += 1e5*s
138:             return tol
139: 
140:         nk = []
141:         for k in range(1,6):
142:             tck = splrep(x,v,s=s,per=per,k=k,xe=xe)
143:             nk.append([splint(ia,ib,tck),spalde(dx,tck)])
144:         put("\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]" % (f(None),
145:                                                    repr(round(xb,3)),repr(round(xe,3)),
146:                                                     repr(round(a,3)),repr(round(b,3))))
147:         put(" per=%d s=%s N=%d [a, b] = [%s, %s]  dx=%s" % (per,repr(s),N,repr(round(ia,3)),repr(round(ib,3)),repr(round(dx,3))))
148:         put(" k :  int(s,[a,b]) Int.Error   Rel. error of s^(d)(dx) d = 0, .., k")
149:         k = 1
150:         for r in nk:
151:             if r[0] < 0:
152:                 sr = '-'
153:             else:
154:                 sr = ' '
155:             put(" %d   %s%.8f   %.1e " % (k,sr,abs(r[0]),
156:                                          abs(r[0]-(f(ib,-1)-f(ia,-1)))))
157:             d = 0
158:             for dr in r[1]:
159:                 err = abs(1-dr/f(dx,d))
160:                 tol = err_est(k, d)
161:                 assert_(err < tol, (k, d))
162:                 put(" %.1e %.1e" % (err, tol))
163:                 d = d+1
164:             put("\n")
165:             k = k+1
166: 
167:     def check_3(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
168:               ia=0,ib=2*pi,dx=0.2*pi):
169:         if xb is None:
170:             xb = a
171:         if xe is None:
172:             xe = b
173:         x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
174:         v = f(x)
175:         put("  k  :     Roots of s(x) approx %s  x in [%s,%s]:" %
176:               (f(None),repr(round(a,3)),repr(round(b,3))))
177:         for k in range(1,6):
178:             tck = splrep(x, v, s=s, per=per, k=k, xe=xe)
179:             if k == 3:
180:                 roots = sproot(tck)
181:                 assert_allclose(splev(roots, tck), 0, atol=1e-10, rtol=1e-10)
182:                 assert_allclose(roots, pi*array([1, 2, 3, 4]), rtol=1e-3)
183:                 put('  %d  : %s' % (k, repr(roots.tolist())))
184:             else:
185:                 assert_raises(ValueError, sproot, tck)
186: 
187:     def check_4(self,f=f1,per=0,s=0,a=0,b=2*pi,N=20,xb=None,xe=None,
188:               ia=0,ib=2*pi,dx=0.2*pi):
189:         if xb is None:
190:             xb = a
191:         if xe is None:
192:             xe = b
193:         x = a+(b-a)*arange(N+1,dtype=float)/float(N)    # nodes
194:         x1 = a + (b-a)*arange(1,N,dtype=float)/float(N-1)  # middle points of the nodes
195:         v,v1 = f(x),f(x1)
196:         put(" u = %s   N = %d" % (repr(round(dx,3)),N))
197:         put("  k  :  [x(u), %s(x(u))]  Error of splprep  Error of splrep " % (f(0,None)))
198:         for k in range(1,6):
199:             tckp,u = splprep([x,v],s=s,per=per,k=k,nest=-1)
200:             tck = splrep(x,v,s=s,per=per,k=k)
201:             uv = splev(dx,tckp)
202:             err1 = abs(uv[1]-f(uv[0]))
203:             err2 = abs(splev(uv[0],tck)-f(uv[0]))
204:             assert_(err1 < 1e-2)
205:             assert_(err2 < 1e-2)
206:             put("  %d  :  %s    %.1e           %.1e" %
207:                   (k,repr([round(z,3) for z in uv]),
208:                    err1,
209:                    err2))
210:         put("Derivatives of parametric cubic spline at u (first function):")
211:         k = 3
212:         tckp,u = splprep([x,v],s=s,per=per,k=k,nest=-1)
213:         for d in range(1,k+1):
214:             uv = splev(dx,tckp,d)
215:             put(" %s " % (repr(uv[0])))
216: 
217:     def check_5(self,f=f2,kx=3,ky=3,xb=0,xe=2*pi,yb=0,ye=2*pi,Nx=20,Ny=20,s=0):
218:         x = xb+(xe-xb)*arange(Nx+1,dtype=float)/float(Nx)
219:         y = yb+(ye-yb)*arange(Ny+1,dtype=float)/float(Ny)
220:         xy = makepairs(x,y)
221:         tck = bisplrep(xy[0],xy[1],f(xy[0],xy[1]),s=s,kx=kx,ky=ky)
222:         tt = [tck[0][kx:-kx],tck[1][ky:-ky]]
223:         t2 = makepairs(tt[0],tt[1])
224:         v1 = bisplev(tt[0],tt[1],tck)
225:         v2 = f2(t2[0],t2[1])
226:         v2.shape = len(tt[0]),len(tt[1])
227:         err = norm2(ravel(v1-v2))
228:         assert_(err < 1e-2, err)
229:         put(err)
230: 
231:     def test_smoke_splrep_splev(self):
232:         put("***************** splrep/splev")
233:         self.check_1(s=1e-6)
234:         self.check_1()
235:         self.check_1(at=1)
236:         self.check_1(per=1)
237:         self.check_1(per=1,at=1)
238:         self.check_1(b=1.5*pi)
239:         self.check_1(b=1.5*pi,xe=2*pi,per=1,s=1e-1)
240: 
241:     def test_smoke_splint_spalde(self):
242:         put("***************** splint/spalde")
243:         self.check_2()
244:         self.check_2(per=1)
245:         self.check_2(ia=0.2*pi,ib=pi)
246:         self.check_2(ia=0.2*pi,ib=pi,N=50)
247: 
248:     def test_smoke_sproot(self):
249:         put("***************** sproot")
250:         self.check_3(a=0.1,b=15)
251: 
252:     def test_smoke_splprep_splrep_splev(self):
253:         put("***************** splprep/splrep/splev")
254:         self.check_4()
255:         self.check_4(N=50)
256: 
257:     def test_smoke_bisplrep_bisplev(self):
258:         put("***************** bisplev")
259:         self.check_5()
260: 
261: 
262: class TestSplev(object):
263:     def test_1d_shape(self):
264:         x = [1,2,3,4,5]
265:         y = [4,5,6,7,8]
266:         tck = splrep(x, y)
267:         z = splev([1], tck)
268:         assert_equal(z.shape, (1,))
269:         z = splev(1, tck)
270:         assert_equal(z.shape, ())
271: 
272:     def test_2d_shape(self):
273:         x = [1, 2, 3, 4, 5]
274:         y = [4, 5, 6, 7, 8]
275:         tck = splrep(x, y)
276:         t = np.array([[1.0, 1.5, 2.0, 2.5],
277:                       [3.0, 3.5, 4.0, 4.5]])
278:         z = splev(t, tck)
279:         z0 = splev(t[0], tck)
280:         z1 = splev(t[1], tck)
281:         assert_equal(z, np.row_stack((z0, z1)))
282: 
283:     def test_extrapolation_modes(self):
284:         # test extrapolation modes
285:         #    * if ext=0, return the extrapolated value.
286:         #    * if ext=1, return 0
287:         #    * if ext=2, raise a ValueError
288:         #    * if ext=3, return the boundary value.
289:         x = [1,2,3]
290:         y = [0,2,4]
291:         tck = splrep(x, y, k=1)
292: 
293:         rstl = [[-2, 6], [0, 0], None, [0, 4]]
294:         for ext in (0, 1, 3):
295:             assert_array_almost_equal(splev([0, 4], tck, ext=ext), rstl[ext])
296: 
297:         assert_raises(ValueError, splev, [0, 4], tck, ext=2)
298: 
299: 
300: class TestSplder(object):
301:     def setup_method(self):
302:         # non-uniform grid, just to make it sure
303:         x = np.linspace(0, 1, 100)**3
304:         y = np.sin(20 * x)
305:         self.spl = splrep(x, y)
306: 
307:         # double check that knots are non-uniform
308:         assert_(np.diff(self.spl[0]).ptp() > 0)
309: 
310:     def test_inverse(self):
311:         # Check that antiderivative + derivative is identity.
312:         for n in range(5):
313:             spl2 = splantider(self.spl, n)
314:             spl3 = splder(spl2, n)
315:             assert_allclose(self.spl[0], spl3[0])
316:             assert_allclose(self.spl[1], spl3[1])
317:             assert_equal(self.spl[2], spl3[2])
318: 
319:     def test_splder_vs_splev(self):
320:         # Check derivative vs. FITPACK
321: 
322:         for n in range(3+1):
323:             # Also extrapolation!
324:             xx = np.linspace(-1, 2, 2000)
325:             if n == 3:
326:                 # ... except that FITPACK extrapolates strangely for
327:                 # order 0, so let's not check that.
328:                 xx = xx[(xx >= 0) & (xx <= 1)]
329: 
330:             dy = splev(xx, self.spl, n)
331:             spl2 = splder(self.spl, n)
332:             dy2 = splev(xx, spl2)
333:             if n == 1:
334:                 assert_allclose(dy, dy2, rtol=2e-6)
335:             else:
336:                 assert_allclose(dy, dy2)
337: 
338:     def test_splantider_vs_splint(self):
339:         # Check antiderivative vs. FITPACK
340:         spl2 = splantider(self.spl)
341: 
342:         # no extrapolation, splint assumes function is zero outside
343:         # range
344:         xx = np.linspace(0, 1, 20)
345: 
346:         for x1 in xx:
347:             for x2 in xx:
348:                 y1 = splint(x1, x2, self.spl)
349:                 y2 = splev(x2, spl2) - splev(x1, spl2)
350:                 assert_allclose(y1, y2)
351: 
352:     def test_order0_diff(self):
353:         assert_raises(ValueError, splder, self.spl, 4)
354: 
355:     def test_kink(self):
356:         # Should refuse to differentiate splines with kinks
357: 
358:         spl2 = insert(0.5, self.spl, m=2)
359:         splder(spl2, 2)  # Should work
360:         assert_raises(ValueError, splder, spl2, 3)
361: 
362:         spl2 = insert(0.5, self.spl, m=3)
363:         splder(spl2, 1)  # Should work
364:         assert_raises(ValueError, splder, spl2, 2)
365: 
366:         spl2 = insert(0.5, self.spl, m=4)
367:         assert_raises(ValueError, splder, spl2, 1)
368: 
369:     def test_multidim(self):
370:         # c can have trailing dims
371:         for n in range(3):
372:             t, c, k = self.spl
373:             c2 = np.c_[c, c, c]
374:             c2 = np.dstack((c2, c2))
375: 
376:             spl2 = splantider((t, c2, k), n)
377:             spl3 = splder(spl2, n)
378: 
379:             assert_allclose(t, spl3[0])
380:             assert_allclose(c2, spl3[1])
381:             assert_equal(k, spl3[2])
382: 
383: 
384: class TestBisplrep(object):
385:     def test_overflow(self):
386:         a = np.linspace(0, 1, 620)
387:         b = np.linspace(0, 1, 620)
388:         x, y = np.meshgrid(a, b)
389:         z = np.random.rand(*x.shape)
390:         assert_raises(OverflowError, bisplrep, x.ravel(), y.ravel(), z.ravel(), s=0)
391: 
392:     def test_regression_1310(self):
393:         # Regression test for gh-1310
394:         data = np.load(data_file('bug-1310.npz'))['data']
395: 
396:         # Shouldn't crash -- the input data triggers work array sizes
397:         # that caused previously some data to not be aligned on
398:         # sizeof(double) boundaries in memory, which made the Fortran
399:         # code to crash when compiled with -O3
400:         bisplrep(data[:,0], data[:,1], data[:,2], kx=3, ky=3, s=0,
401:                  full_output=True)
402: 
403: 
404: def test_dblint():
405:     # Basic test to see it runs and gives the correct result on a trivial
406:     # problem.  Note that `dblint` is not exposed in the interpolate namespace.
407:     x = np.linspace(0, 1)
408:     y = np.linspace(0, 1)
409:     xx, yy = np.meshgrid(x, y)
410:     rect = interpolate.RectBivariateSpline(x, y, 4 * xx * yy)
411:     tck = list(rect.tck)
412:     tck.extend(rect.degrees)
413: 
414:     assert_almost_equal(dblint(0, 1, 0, 1, tck), 1)
415:     assert_almost_equal(dblint(0, 0.5, 0, 1, tck), 0.25)
416:     assert_almost_equal(dblint(0.5, 1, 0, 1, tck), 0.75)
417:     assert_almost_equal(dblint(-100, 100, -100, 100, tck), 1)
418: 
419: 
420: def test_splev_der_k():
421:     # regression test for gh-2188: splev(x, tck, der=k) gives garbage or crashes
422:     # for x outside of knot range
423: 
424:     # test case from gh-2188
425:     tck = (np.array([0., 0., 2.5, 2.5]),
426:            np.array([-1.56679978, 2.43995873, 0., 0.]),
427:            1)
428:     t, c, k = tck
429:     x = np.array([-3, 0, 2.5, 3])
430: 
431:     # an explicit form of the linear spline
432:     assert_allclose(splev(x, tck), c[0] + (c[1] - c[0]) * x/t[2])
433:     assert_allclose(splev(x, tck, 1), (c[1]-c[0]) / t[2])
434: 
435:     # now check a random spline vs splder
436:     np.random.seed(1234)
437:     x = np.sort(np.random.random(30))
438:     y = np.random.random(30)
439:     t, c, k = splrep(x, y)
440: 
441:     x = [t[0] - 1., t[-1] + 1.]
442:     tck2 = splder((t, c, k), k)
443:     assert_allclose(splev(x, (t, c, k), k), splev(x, tck2))
444: 
445: 
446: def test_bisplev_integer_overflow():
447:     np.random.seed(1)
448: 
449:     x = np.linspace(0, 1, 11)
450:     y = x
451:     z = np.random.randn(11, 11).ravel()
452:     kx = 1
453:     ky = 1
454: 
455:     nx, tx, ny, ty, c, fp, ier = regrid_smth(
456:         x, y, z, None, None, None, None, kx=kx, ky=ky, s=0.0)
457:     tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)], kx, ky)
458: 
459:     xp = np.zeros([2621440])
460:     yp = np.zeros([2621440])
461: 
462:     assert_raises((RuntimeError, MemoryError), bisplev, xp, yp, tck)
463: 
464: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_88976) is not StypyTypeError):

    if (import_88976 != 'pyd_module'):
        __import__(import_88976)
        sys_modules_88977 = sys.modules[import_88976]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_88977.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_88976)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_equal, assert_allclose, assert_, assert_almost_equal, assert_array_almost_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_88978) is not StypyTypeError):

    if (import_88978 != 'pyd_module'):
        __import__(import_88978)
        sys_modules_88979 = sys.modules[import_88978]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_88979.module_type_store, module_type_store, ['assert_equal', 'assert_allclose', 'assert_', 'assert_almost_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_88979, sys_modules_88979.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose, assert_, assert_almost_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose', 'assert_', 'assert_almost_equal', 'assert_array_almost_equal'], [assert_equal, assert_allclose, assert_, assert_almost_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_88978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_88980) is not StypyTypeError):

    if (import_88980 != 'pyd_module'):
        __import__(import_88980)
        sys_modules_88981 = sys.modules[import_88980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_88981.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_88981, sys_modules_88981.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_88980)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy import array, asarray, pi, sin, cos, arange, dot, ravel, sqrt, round' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_88982) is not StypyTypeError):

    if (import_88982 != 'pyd_module'):
        __import__(import_88982)
        sys_modules_88983 = sys.modules[import_88982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', sys_modules_88983.module_type_store, module_type_store, ['array', 'asarray', 'pi', 'sin', 'cos', 'arange', 'dot', 'ravel', 'sqrt', 'round'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_88983, sys_modules_88983.module_type_store, module_type_store)
    else:
        from numpy import array, asarray, pi, sin, cos, arange, dot, ravel, sqrt, round

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', None, module_type_store, ['array', 'asarray', 'pi', 'sin', 'cos', 'arange', 'dot', 'ravel', 'sqrt', 'round'], [array, asarray, pi, sin, cos, arange, dot, ravel, sqrt, round])

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_88982)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy import interpolate' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88984 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy')

if (type(import_88984) is not StypyTypeError):

    if (import_88984 != 'pyd_module'):
        __import__(import_88984)
        sys_modules_88985 = sys.modules[import_88984]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', sys_modules_88985.module_type_store, module_type_store, ['interpolate'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_88985, sys_modules_88985.module_type_store, module_type_store)
    else:
        from scipy import interpolate

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', None, module_type_store, ['interpolate'], [interpolate])

else:
    # Assigning a type to the variable 'scipy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy', import_88984)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.interpolate.fitpack import splrep, splev, bisplrep, bisplev, sproot, splprep, splint, spalde, splder, splantider, insert, dblint' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate.fitpack')

if (type(import_88986) is not StypyTypeError):

    if (import_88986 != 'pyd_module'):
        __import__(import_88986)
        sys_modules_88987 = sys.modules[import_88986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate.fitpack', sys_modules_88987.module_type_store, module_type_store, ['splrep', 'splev', 'bisplrep', 'bisplev', 'sproot', 'splprep', 'splint', 'spalde', 'splder', 'splantider', 'insert', 'dblint'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_88987, sys_modules_88987.module_type_store, module_type_store)
    else:
        from scipy.interpolate.fitpack import splrep, splev, bisplrep, bisplev, sproot, splprep, splint, spalde, splder, splantider, insert, dblint

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate.fitpack', None, module_type_store, ['splrep', 'splev', 'bisplrep', 'bisplev', 'sproot', 'splprep', 'splint', 'spalde', 'splder', 'splantider', 'insert', 'dblint'], [splrep, splev, bisplrep, bisplev, sproot, splprep, splint, spalde, splder, splantider, insert, dblint])

else:
    # Assigning a type to the variable 'scipy.interpolate.fitpack' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.interpolate.fitpack', import_88986)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.interpolate.dfitpack import regrid_smth' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_88988 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.interpolate.dfitpack')

if (type(import_88988) is not StypyTypeError):

    if (import_88988 != 'pyd_module'):
        __import__(import_88988)
        sys_modules_88989 = sys.modules[import_88988]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.interpolate.dfitpack', sys_modules_88989.module_type_store, module_type_store, ['regrid_smth'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_88989, sys_modules_88989.module_type_store, module_type_store)
    else:
        from scipy.interpolate.dfitpack import regrid_smth

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.interpolate.dfitpack', None, module_type_store, ['regrid_smth'], [regrid_smth])

else:
    # Assigning a type to the variable 'scipy.interpolate.dfitpack' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.interpolate.dfitpack', import_88988)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')


@norecursion
def data_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'data_file'
    module_type_store = module_type_store.open_function_context('data_file', 17, 0, False)
    
    # Passed parameters checking function
    data_file.stypy_localization = localization
    data_file.stypy_type_of_self = None
    data_file.stypy_type_store = module_type_store
    data_file.stypy_function_name = 'data_file'
    data_file.stypy_param_names_list = ['basename']
    data_file.stypy_varargs_param_name = None
    data_file.stypy_kwargs_param_name = None
    data_file.stypy_call_defaults = defaults
    data_file.stypy_call_varargs = varargs
    data_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'data_file', ['basename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'data_file', localization, ['basename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'data_file(...)' code ##################

    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to abspath(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to dirname(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of '__file__' (line 18)
    file___88999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 56), '__file__', False)
    # Processing the call keyword arguments (line 18)
    kwargs_89000 = {}
    # Getting the type of 'os' (line 18)
    os_88996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 40), 'os', False)
    # Obtaining the member 'path' of a type (line 18)
    path_88997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 40), os_88996, 'path')
    # Obtaining the member 'dirname' of a type (line 18)
    dirname_88998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 40), path_88997, 'dirname')
    # Calling dirname(args, kwargs) (line 18)
    dirname_call_result_89001 = invoke(stypy.reporting.localization.Localization(__file__, 18, 40), dirname_88998, *[file___88999], **kwargs_89000)
    
    # Processing the call keyword arguments (line 18)
    kwargs_89002 = {}
    # Getting the type of 'os' (line 18)
    os_88993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 18)
    path_88994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), os_88993, 'path')
    # Obtaining the member 'abspath' of a type (line 18)
    abspath_88995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 24), path_88994, 'abspath')
    # Calling abspath(args, kwargs) (line 18)
    abspath_call_result_89003 = invoke(stypy.reporting.localization.Localization(__file__, 18, 24), abspath_88995, *[dirname_call_result_89001], **kwargs_89002)
    
    str_89004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 24), 'str', 'data')
    # Getting the type of 'basename' (line 19)
    basename_89005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'basename', False)
    # Processing the call keyword arguments (line 18)
    kwargs_89006 = {}
    # Getting the type of 'os' (line 18)
    os_88990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 18)
    path_88991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), os_88990, 'path')
    # Obtaining the member 'join' of a type (line 18)
    join_88992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), path_88991, 'join')
    # Calling join(args, kwargs) (line 18)
    join_call_result_89007 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), join_88992, *[abspath_call_result_89003, str_89004, basename_89005], **kwargs_89006)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', join_call_result_89007)
    
    # ################# End of 'data_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'data_file' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_89008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89008)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'data_file'
    return stypy_return_type_89008

# Assigning a type to the variable 'data_file' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'data_file', data_file)

@norecursion
def norm2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'norm2'
    module_type_store = module_type_store.open_function_context('norm2', 22, 0, False)
    
    # Passed parameters checking function
    norm2.stypy_localization = localization
    norm2.stypy_type_of_self = None
    norm2.stypy_type_store = module_type_store
    norm2.stypy_function_name = 'norm2'
    norm2.stypy_param_names_list = ['x']
    norm2.stypy_varargs_param_name = None
    norm2.stypy_kwargs_param_name = None
    norm2.stypy_call_defaults = defaults
    norm2.stypy_call_varargs = varargs
    norm2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'norm2', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'norm2', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'norm2(...)' code ##################

    
    # Call to sqrt(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to dot(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'x' (line 23)
    x_89011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'x', False)
    # Obtaining the member 'T' of a type (line 23)
    T_89012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), x_89011, 'T')
    # Getting the type of 'x' (line 23)
    x_89013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'x', False)
    # Processing the call keyword arguments (line 23)
    kwargs_89014 = {}
    # Getting the type of 'dot' (line 23)
    dot_89010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'dot', False)
    # Calling dot(args, kwargs) (line 23)
    dot_call_result_89015 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), dot_89010, *[T_89012, x_89013], **kwargs_89014)
    
    # Processing the call keyword arguments (line 23)
    kwargs_89016 = {}
    # Getting the type of 'sqrt' (line 23)
    sqrt_89009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 23)
    sqrt_call_result_89017 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), sqrt_89009, *[dot_call_result_89015], **kwargs_89016)
    
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', sqrt_call_result_89017)
    
    # ################# End of 'norm2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norm2' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_89018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89018)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norm2'
    return stypy_return_type_89018

# Assigning a type to the variable 'norm2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'norm2', norm2)

@norecursion
def f1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_89019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'int')
    defaults = [int_89019]
    # Create a new context for function 'f1'
    module_type_store = module_type_store.open_function_context('f1', 26, 0, False)
    
    # Passed parameters checking function
    f1.stypy_localization = localization
    f1.stypy_type_of_self = None
    f1.stypy_type_store = module_type_store
    f1.stypy_function_name = 'f1'
    f1.stypy_param_names_list = ['x', 'd']
    f1.stypy_varargs_param_name = None
    f1.stypy_kwargs_param_name = None
    f1.stypy_call_defaults = defaults
    f1.stypy_call_varargs = varargs
    f1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f1', ['x', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f1', localization, ['x', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f1(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 27)
    # Getting the type of 'd' (line 27)
    d_89020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'd')
    # Getting the type of 'None' (line 27)
    None_89021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'None')
    
    (may_be_89022, more_types_in_union_89023) = may_be_none(d_89020, None_89021)

    if may_be_89022:

        if more_types_in_union_89023:
            # Runtime conditional SSA (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        str_89024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'str', 'sin')
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', str_89024)

        if more_types_in_union_89023:
            # SSA join for if statement (line 27)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 29)
    # Getting the type of 'x' (line 29)
    x_89025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'x')
    # Getting the type of 'None' (line 29)
    None_89026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'None')
    
    (may_be_89027, more_types_in_union_89028) = may_be_none(x_89025, None_89026)

    if may_be_89027:

        if more_types_in_union_89028:
            # Runtime conditional SSA (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        str_89029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'str', 'sin(x)')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', str_89029)

        if more_types_in_union_89028:
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'd' (line 31)
    d_89030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'd')
    int_89031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'int')
    # Applying the binary operator '%' (line 31)
    result_mod_89032 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '%', d_89030, int_89031)
    
    int_89033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'int')
    # Applying the binary operator '==' (line 31)
    result_eq_89034 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '==', result_mod_89032, int_89033)
    
    # Testing the type of an if condition (line 31)
    if_condition_89035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), result_eq_89034)
    # Assigning a type to the variable 'if_condition_89035' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_89035', if_condition_89035)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sin(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'x' (line 32)
    x_89037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'x', False)
    # Processing the call keyword arguments (line 32)
    kwargs_89038 = {}
    # Getting the type of 'sin' (line 32)
    sin_89036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'sin', False)
    # Calling sin(args, kwargs) (line 32)
    sin_call_result_89039 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), sin_89036, *[x_89037], **kwargs_89038)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', sin_call_result_89039)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 33)
    d_89040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'd')
    int_89041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'int')
    # Applying the binary operator '%' (line 33)
    result_mod_89042 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), '%', d_89040, int_89041)
    
    int_89043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'int')
    # Applying the binary operator '==' (line 33)
    result_eq_89044 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 7), '==', result_mod_89042, int_89043)
    
    # Testing the type of an if condition (line 33)
    if_condition_89045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 4), result_eq_89044)
    # Assigning a type to the variable 'if_condition_89045' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'if_condition_89045', if_condition_89045)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cos(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'x' (line 34)
    x_89047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'x', False)
    # Processing the call keyword arguments (line 34)
    kwargs_89048 = {}
    # Getting the type of 'cos' (line 34)
    cos_89046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'cos', False)
    # Calling cos(args, kwargs) (line 34)
    cos_call_result_89049 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), cos_89046, *[x_89047], **kwargs_89048)
    
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', cos_call_result_89049)
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 35)
    d_89050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'd')
    int_89051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'int')
    # Applying the binary operator '%' (line 35)
    result_mod_89052 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 7), '%', d_89050, int_89051)
    
    int_89053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'int')
    # Applying the binary operator '==' (line 35)
    result_eq_89054 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 7), '==', result_mod_89052, int_89053)
    
    # Testing the type of an if condition (line 35)
    if_condition_89055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), result_eq_89054)
    # Assigning a type to the variable 'if_condition_89055' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_89055', if_condition_89055)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to sin(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'x' (line 36)
    x_89057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'x', False)
    # Processing the call keyword arguments (line 36)
    kwargs_89058 = {}
    # Getting the type of 'sin' (line 36)
    sin_89056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'sin', False)
    # Calling sin(args, kwargs) (line 36)
    sin_call_result_89059 = invoke(stypy.reporting.localization.Localization(__file__, 36, 16), sin_89056, *[x_89057], **kwargs_89058)
    
    # Applying the 'usub' unary operator (line 36)
    result___neg___89060 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), 'usub', sin_call_result_89059)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', result___neg___89060)
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 37)
    d_89061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'd')
    int_89062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'int')
    # Applying the binary operator '%' (line 37)
    result_mod_89063 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), '%', d_89061, int_89062)
    
    int_89064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'int')
    # Applying the binary operator '==' (line 37)
    result_eq_89065 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), '==', result_mod_89063, int_89064)
    
    # Testing the type of an if condition (line 37)
    if_condition_89066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_eq_89065)
    # Assigning a type to the variable 'if_condition_89066' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_89066', if_condition_89066)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to cos(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_89068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'x', False)
    # Processing the call keyword arguments (line 38)
    kwargs_89069 = {}
    # Getting the type of 'cos' (line 38)
    cos_89067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'cos', False)
    # Calling cos(args, kwargs) (line 38)
    cos_call_result_89070 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), cos_89067, *[x_89068], **kwargs_89069)
    
    # Applying the 'usub' unary operator (line 38)
    result___neg___89071 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 15), 'usub', cos_call_result_89070)
    
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', result___neg___89071)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'f1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f1' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_89072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89072)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f1'
    return stypy_return_type_89072

# Assigning a type to the variable 'f1' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'f1', f1)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_89073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'int')
    int_89074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'int')
    int_89075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    defaults = [int_89073, int_89074, int_89075]
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 41, 0, False)
    
    # Passed parameters checking function
    f2.stypy_localization = localization
    f2.stypy_type_of_self = None
    f2.stypy_type_store = module_type_store
    f2.stypy_function_name = 'f2'
    f2.stypy_param_names_list = ['x', 'y', 'dx', 'dy']
    f2.stypy_varargs_param_name = None
    f2.stypy_kwargs_param_name = None
    f2.stypy_call_defaults = defaults
    f2.stypy_call_varargs = varargs
    f2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f2', ['x', 'y', 'dx', 'dy'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f2', localization, ['x', 'y', 'dx', 'dy'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f2(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'x' (line 42)
    x_89076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'x')
    # Getting the type of 'None' (line 42)
    None_89077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'None')
    
    (may_be_89078, more_types_in_union_89079) = may_be_none(x_89076, None_89077)

    if may_be_89078:

        if more_types_in_union_89079:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        str_89080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 15), 'str', 'sin(x+y)')
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', str_89080)

        if more_types_in_union_89079:
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 44):
    
    # Assigning a BinOp to a Name (line 44):
    # Getting the type of 'dx' (line 44)
    dx_89081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'dx')
    # Getting the type of 'dy' (line 44)
    dy_89082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'dy')
    # Applying the binary operator '+' (line 44)
    result_add_89083 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 8), '+', dx_89081, dy_89082)
    
    # Assigning a type to the variable 'd' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'd', result_add_89083)
    
    
    # Getting the type of 'd' (line 45)
    d_89084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 7), 'd')
    int_89085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 11), 'int')
    # Applying the binary operator '%' (line 45)
    result_mod_89086 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), '%', d_89084, int_89085)
    
    int_89087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'int')
    # Applying the binary operator '==' (line 45)
    result_eq_89088 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 7), '==', result_mod_89086, int_89087)
    
    # Testing the type of an if condition (line 45)
    if_condition_89089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_eq_89088)
    # Assigning a type to the variable 'if_condition_89089' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'if_condition_89089', if_condition_89089)
    # SSA begins for if statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to sin(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'x' (line 46)
    x_89091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'x', False)
    # Getting the type of 'y' (line 46)
    y_89092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'y', False)
    # Applying the binary operator '+' (line 46)
    result_add_89093 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 19), '+', x_89091, y_89092)
    
    # Processing the call keyword arguments (line 46)
    kwargs_89094 = {}
    # Getting the type of 'sin' (line 46)
    sin_89090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'sin', False)
    # Calling sin(args, kwargs) (line 46)
    sin_call_result_89095 = invoke(stypy.reporting.localization.Localization(__file__, 46, 15), sin_89090, *[result_add_89093], **kwargs_89094)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', sin_call_result_89095)
    # SSA join for if statement (line 45)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 47)
    d_89096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'd')
    int_89097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
    # Applying the binary operator '%' (line 47)
    result_mod_89098 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), '%', d_89096, int_89097)
    
    int_89099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'int')
    # Applying the binary operator '==' (line 47)
    result_eq_89100 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), '==', result_mod_89098, int_89099)
    
    # Testing the type of an if condition (line 47)
    if_condition_89101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_eq_89100)
    # Assigning a type to the variable 'if_condition_89101' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_89101', if_condition_89101)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to cos(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'x' (line 48)
    x_89103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'x', False)
    # Getting the type of 'y' (line 48)
    y_89104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'y', False)
    # Applying the binary operator '+' (line 48)
    result_add_89105 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '+', x_89103, y_89104)
    
    # Processing the call keyword arguments (line 48)
    kwargs_89106 = {}
    # Getting the type of 'cos' (line 48)
    cos_89102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'cos', False)
    # Calling cos(args, kwargs) (line 48)
    cos_call_result_89107 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), cos_89102, *[result_add_89105], **kwargs_89106)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', cos_call_result_89107)
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 49)
    d_89108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 7), 'd')
    int_89109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'int')
    # Applying the binary operator '%' (line 49)
    result_mod_89110 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), '%', d_89108, int_89109)
    
    int_89111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'int')
    # Applying the binary operator '==' (line 49)
    result_eq_89112 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), '==', result_mod_89110, int_89111)
    
    # Testing the type of an if condition (line 49)
    if_condition_89113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_eq_89112)
    # Assigning a type to the variable 'if_condition_89113' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_89113', if_condition_89113)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to sin(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'x' (line 50)
    x_89115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'x', False)
    # Getting the type of 'y' (line 50)
    y_89116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'y', False)
    # Applying the binary operator '+' (line 50)
    result_add_89117 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 20), '+', x_89115, y_89116)
    
    # Processing the call keyword arguments (line 50)
    kwargs_89118 = {}
    # Getting the type of 'sin' (line 50)
    sin_89114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'sin', False)
    # Calling sin(args, kwargs) (line 50)
    sin_call_result_89119 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), sin_89114, *[result_add_89117], **kwargs_89118)
    
    # Applying the 'usub' unary operator (line 50)
    result___neg___89120 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'usub', sin_call_result_89119)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', result___neg___89120)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 51)
    d_89121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'd')
    int_89122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'int')
    # Applying the binary operator '%' (line 51)
    result_mod_89123 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '%', d_89121, int_89122)
    
    int_89124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'int')
    # Applying the binary operator '==' (line 51)
    result_eq_89125 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '==', result_mod_89123, int_89124)
    
    # Testing the type of an if condition (line 51)
    if_condition_89126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), result_eq_89125)
    # Assigning a type to the variable 'if_condition_89126' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_89126', if_condition_89126)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to cos(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'x' (line 52)
    x_89128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'x', False)
    # Getting the type of 'y' (line 52)
    y_89129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'y', False)
    # Applying the binary operator '+' (line 52)
    result_add_89130 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '+', x_89128, y_89129)
    
    # Processing the call keyword arguments (line 52)
    kwargs_89131 = {}
    # Getting the type of 'cos' (line 52)
    cos_89127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'cos', False)
    # Calling cos(args, kwargs) (line 52)
    cos_call_result_89132 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), cos_89127, *[result_add_89130], **kwargs_89131)
    
    # Applying the 'usub' unary operator (line 52)
    result___neg___89133 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 15), 'usub', cos_call_result_89132)
    
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', result___neg___89133)
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_89134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89134)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_89134

# Assigning a type to the variable 'f2' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'f2', f2)

@norecursion
def makepairs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'makepairs'
    module_type_store = module_type_store.open_function_context('makepairs', 55, 0, False)
    
    # Passed parameters checking function
    makepairs.stypy_localization = localization
    makepairs.stypy_type_of_self = None
    makepairs.stypy_type_store = module_type_store
    makepairs.stypy_function_name = 'makepairs'
    makepairs.stypy_param_names_list = ['x', 'y']
    makepairs.stypy_varargs_param_name = None
    makepairs.stypy_kwargs_param_name = None
    makepairs.stypy_call_defaults = defaults
    makepairs.stypy_call_varargs = varargs
    makepairs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'makepairs', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'makepairs', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'makepairs(...)' code ##################

    str_89135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'Helper function to create an array of pairs of x and y.')
    
    # Assigning a Call to a Name (line 58):
    
    # Assigning a Call to a Name (line 58):
    
    # Call to array(...): (line 58)
    # Processing the call arguments (line 58)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to asarray(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'x' (line 58)
    x_89141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'x', False)
    # Processing the call keyword arguments (line 58)
    kwargs_89142 = {}
    # Getting the type of 'asarray' (line 58)
    asarray_89140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'asarray', False)
    # Calling asarray(args, kwargs) (line 58)
    asarray_call_result_89143 = invoke(stypy.reporting.localization.Localization(__file__, 58, 32), asarray_89140, *[x_89141], **kwargs_89142)
    
    comprehension_89144 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), asarray_call_result_89143)
    # Assigning a type to the variable 'a' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'a', comprehension_89144)
    # Calculating comprehension expression
    
    # Call to asarray(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'y' (line 58)
    y_89146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 60), 'y', False)
    # Processing the call keyword arguments (line 58)
    kwargs_89147 = {}
    # Getting the type of 'asarray' (line 58)
    asarray_89145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'asarray', False)
    # Calling asarray(args, kwargs) (line 58)
    asarray_call_result_89148 = invoke(stypy.reporting.localization.Localization(__file__, 58, 52), asarray_89145, *[y_89146], **kwargs_89147)
    
    comprehension_89149 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), asarray_call_result_89148)
    # Assigning a type to the variable 'b' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'b', comprehension_89149)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_89137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    # Getting the type of 'a' (line 58)
    a_89138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 'a', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), list_89137, a_89138)
    # Adding element type (line 58)
    # Getting the type of 'b' (line 58)
    b_89139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'b', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), list_89137, b_89139)
    
    list_89150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 16), list_89150, list_89137)
    # Processing the call keyword arguments (line 58)
    kwargs_89151 = {}
    # Getting the type of 'array' (line 58)
    array_89136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'array', False)
    # Calling array(args, kwargs) (line 58)
    array_call_result_89152 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), array_89136, *[list_89150], **kwargs_89151)
    
    # Assigning a type to the variable 'xy' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'xy', array_call_result_89152)
    # Getting the type of 'xy' (line 59)
    xy_89153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'xy')
    # Obtaining the member 'T' of a type (line 59)
    T_89154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), xy_89153, 'T')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', T_89154)
    
    # ################# End of 'makepairs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'makepairs' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_89155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89155)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'makepairs'
    return stypy_return_type_89155

# Assigning a type to the variable 'makepairs' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'makepairs', makepairs)

@norecursion
def put(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'put'
    module_type_store = module_type_store.open_function_context('put', 62, 0, False)
    
    # Passed parameters checking function
    put.stypy_localization = localization
    put.stypy_type_of_self = None
    put.stypy_type_store = module_type_store
    put.stypy_function_name = 'put'
    put.stypy_param_names_list = []
    put.stypy_varargs_param_name = 'a'
    put.stypy_kwargs_param_name = None
    put.stypy_call_defaults = defaults
    put.stypy_call_varargs = varargs
    put.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'put', [], 'a', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'put', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'put(...)' code ##################

    str_89156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'Produce some output if file run directly')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 64, 4))
    
    # 'import sys' statement (line 64)
    import sys

    import_module(stypy.reporting.localization.Localization(__file__, 64, 4), 'sys', sys, module_type_store)
    
    
    # Type idiom detected: calculating its left and rigth part (line 65)
    str_89157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 40), 'str', '__put_prints')
    
    # Obtaining the type of the subscript
    str_89158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'str', '__main__')
    # Getting the type of 'sys' (line 65)
    sys_89159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'sys')
    # Obtaining the member 'modules' of a type (line 65)
    modules_89160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), sys_89159, 'modules')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___89161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), modules_89160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_89162 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), getitem___89161, str_89158)
    
    
    (may_be_89163, more_types_in_union_89164) = may_provide_member(str_89157, subscript_call_result_89162)

    if may_be_89163:

        if more_types_in_union_89164:
            # Runtime conditional SSA (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to write(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to join(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Call to map(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'str' (line 66)
        str_89171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 37), 'str', False)
        # Getting the type of 'a' (line 66)
        a_89172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'a', False)
        # Processing the call keyword arguments (line 66)
        kwargs_89173 = {}
        # Getting the type of 'map' (line 66)
        map_89170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'map', False)
        # Calling map(args, kwargs) (line 66)
        map_call_result_89174 = invoke(stypy.reporting.localization.Localization(__file__, 66, 33), map_89170, *[str_89171, a_89172], **kwargs_89173)
        
        # Processing the call keyword arguments (line 66)
        kwargs_89175 = {}
        str_89168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', '')
        # Obtaining the member 'join' of a type (line 66)
        join_89169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 25), str_89168, 'join')
        # Calling join(args, kwargs) (line 66)
        join_call_result_89176 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), join_89169, *[map_call_result_89174], **kwargs_89175)
        
        str_89177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'str', '\n')
        # Applying the binary operator '+' (line 66)
        result_add_89178 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 25), '+', join_call_result_89176, str_89177)
        
        # Processing the call keyword arguments (line 66)
        kwargs_89179 = {}
        # Getting the type of 'sys' (line 66)
        sys_89165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 66)
        stderr_89166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), sys_89165, 'stderr')
        # Obtaining the member 'write' of a type (line 66)
        write_89167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), stderr_89166, 'write')
        # Calling write(args, kwargs) (line 66)
        write_call_result_89180 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), write_89167, *[result_add_89178], **kwargs_89179)
        

        if more_types_in_union_89164:
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'put(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'put' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_89181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_89181)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'put'
    return stypy_return_type_89181

# Assigning a type to the variable 'put' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'put', put)
# Declaration of the 'TestSmokeTests' class

class TestSmokeTests(object, ):
    str_89182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, (-1)), 'str', '\n    Smoke tests (with a few asserts) for fitpack routines -- mostly\n    check that they are runnable\n    ')

    @norecursion
    def check_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'f1' (line 75)
        f1_89183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'f1')
        int_89184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 30), 'int')
        int_89185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'int')
        int_89186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'int')
        int_89187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'int')
        # Getting the type of 'pi' (line 75)
        pi_89188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), 'pi')
        # Applying the binary operator '*' (line 75)
        result_mul_89189 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 42), '*', int_89187, pi_89188)
        
        int_89190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 49), 'int')
        int_89191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 55), 'int')
        # Getting the type of 'None' (line 75)
        None_89192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 60), 'None')
        # Getting the type of 'None' (line 75)
        None_89193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 68), 'None')
        defaults = [f1_89183, int_89184, int_89185, int_89186, result_mul_89189, int_89190, int_89191, None_89192, None_89193]
        # Create a new context for function 'check_1'
        module_type_store = module_type_store.open_function_context('check_1', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.check_1')
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_param_names_list', ['f', 'per', 's', 'a', 'b', 'N', 'at', 'xb', 'xe'])
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.check_1.__dict__.__setitem__('stypy_declared_arg_number', 10)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.check_1', ['f', 'per', 's', 'a', 'b', 'N', 'at', 'xb', 'xe'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_1', localization, ['f', 'per', 's', 'a', 'b', 'N', 'at', 'xb', 'xe'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_1(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 76)
        # Getting the type of 'xb' (line 76)
        xb_89194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'xb')
        # Getting the type of 'None' (line 76)
        None_89195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'None')
        
        (may_be_89196, more_types_in_union_89197) = may_be_none(xb_89194, None_89195)

        if may_be_89196:

            if more_types_in_union_89197:
                # Runtime conditional SSA (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 77):
            
            # Assigning a Name to a Name (line 77):
            # Getting the type of 'a' (line 77)
            a_89198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'a')
            # Assigning a type to the variable 'xb' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'xb', a_89198)

            if more_types_in_union_89197:
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 78)
        # Getting the type of 'xe' (line 78)
        xe_89199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'xe')
        # Getting the type of 'None' (line 78)
        None_89200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 17), 'None')
        
        (may_be_89201, more_types_in_union_89202) = may_be_none(xe_89199, None_89200)

        if may_be_89201:

            if more_types_in_union_89202:
                # Runtime conditional SSA (line 78)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 79):
            
            # Assigning a Name to a Name (line 79):
            # Getting the type of 'b' (line 79)
            b_89203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'b')
            # Assigning a type to the variable 'xe' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'xe', b_89203)

            if more_types_in_union_89202:
                # SSA join for if statement (line 78)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 80):
        
        # Assigning a BinOp to a Name (line 80):
        # Getting the type of 'a' (line 80)
        a_89204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'a')
        # Getting the type of 'b' (line 80)
        b_89205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'b')
        # Getting the type of 'a' (line 80)
        a_89206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'a')
        # Applying the binary operator '-' (line 80)
        result_sub_89207 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), '-', b_89205, a_89206)
        
        
        # Call to arange(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'N' (line 80)
        N_89209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 27), 'N', False)
        int_89210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
        # Applying the binary operator '+' (line 80)
        result_add_89211 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 27), '+', N_89209, int_89210)
        
        # Processing the call keyword arguments (line 80)
        # Getting the type of 'float' (line 80)
        float_89212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'float', False)
        keyword_89213 = float_89212
        kwargs_89214 = {'dtype': keyword_89213}
        # Getting the type of 'arange' (line 80)
        arange_89208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 80)
        arange_call_result_89215 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), arange_89208, *[result_add_89211], **kwargs_89214)
        
        # Applying the binary operator '*' (line 80)
        result_mul_89216 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 14), '*', result_sub_89207, arange_call_result_89215)
        
        
        # Call to float(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'N' (line 80)
        N_89218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 50), 'N', False)
        # Processing the call keyword arguments (line 80)
        kwargs_89219 = {}
        # Getting the type of 'float' (line 80)
        float_89217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'float', False)
        # Calling float(args, kwargs) (line 80)
        float_call_result_89220 = invoke(stypy.reporting.localization.Localization(__file__, 80, 44), float_89217, *[N_89218], **kwargs_89219)
        
        # Applying the binary operator 'div' (line 80)
        result_div_89221 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 43), 'div', result_mul_89216, float_call_result_89220)
        
        # Applying the binary operator '+' (line 80)
        result_add_89222 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '+', a_89204, result_div_89221)
        
        # Assigning a type to the variable 'x' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'x', result_add_89222)
        
        # Assigning a BinOp to a Name (line 81):
        
        # Assigning a BinOp to a Name (line 81):
        # Getting the type of 'a' (line 81)
        a_89223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 13), 'a')
        # Getting the type of 'b' (line 81)
        b_89224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'b')
        # Getting the type of 'a' (line 81)
        a_89225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 18), 'a')
        # Applying the binary operator '-' (line 81)
        result_sub_89226 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 16), '-', b_89224, a_89225)
        
        
        # Call to arange(...): (line 81)
        # Processing the call arguments (line 81)
        int_89228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'int')
        # Getting the type of 'N' (line 81)
        N_89229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 30), 'N', False)
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'float' (line 81)
        float_89230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'float', False)
        keyword_89231 = float_89230
        kwargs_89232 = {'dtype': keyword_89231}
        # Getting the type of 'arange' (line 81)
        arange_89227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'arange', False)
        # Calling arange(args, kwargs) (line 81)
        arange_call_result_89233 = invoke(stypy.reporting.localization.Localization(__file__, 81, 21), arange_89227, *[int_89228, N_89229], **kwargs_89232)
        
        # Applying the binary operator '*' (line 81)
        result_mul_89234 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '*', result_sub_89226, arange_call_result_89233)
        
        
        # Call to float(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'N' (line 81)
        N_89236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 51), 'N', False)
        int_89237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 53), 'int')
        # Applying the binary operator '-' (line 81)
        result_sub_89238 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 51), '-', N_89236, int_89237)
        
        # Processing the call keyword arguments (line 81)
        kwargs_89239 = {}
        # Getting the type of 'float' (line 81)
        float_89235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 45), 'float', False)
        # Calling float(args, kwargs) (line 81)
        float_call_result_89240 = invoke(stypy.reporting.localization.Localization(__file__, 81, 45), float_89235, *[result_sub_89238], **kwargs_89239)
        
        # Applying the binary operator 'div' (line 81)
        result_div_89241 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 44), 'div', result_mul_89234, float_call_result_89240)
        
        # Applying the binary operator '+' (line 81)
        result_add_89242 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 13), '+', a_89223, result_div_89241)
        
        # Assigning a type to the variable 'x1' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'x1', result_add_89242)
        
        # Assigning a Tuple to a Tuple (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to f(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'x' (line 82)
        x_89244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'x', False)
        # Processing the call keyword arguments (line 82)
        kwargs_89245 = {}
        # Getting the type of 'f' (line 82)
        f_89243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'f', False)
        # Calling f(args, kwargs) (line 82)
        f_call_result_89246 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), f_89243, *[x_89244], **kwargs_89245)
        
        # Assigning a type to the variable 'tuple_assignment_88948' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_assignment_88948', f_call_result_89246)
        
        # Assigning a Call to a Name (line 82):
        
        # Call to f(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'x1' (line 82)
        x1_89248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'x1', False)
        # Processing the call keyword arguments (line 82)
        kwargs_89249 = {}
        # Getting the type of 'f' (line 82)
        f_89247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'f', False)
        # Calling f(args, kwargs) (line 82)
        f_call_result_89250 = invoke(stypy.reporting.localization.Localization(__file__, 82, 20), f_89247, *[x1_89248], **kwargs_89249)
        
        # Assigning a type to the variable 'tuple_assignment_88949' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_assignment_88949', f_call_result_89250)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_assignment_88948' (line 82)
        tuple_assignment_88948_89251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_assignment_88948')
        # Assigning a type to the variable 'v' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'v', tuple_assignment_88948_89251)
        
        # Assigning a Name to a Name (line 82):
        # Getting the type of 'tuple_assignment_88949' (line 82)
        tuple_assignment_88949_89252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_assignment_88949')
        # Assigning a type to the variable 'v1' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 10), 'v1', tuple_assignment_88949_89252)
        
        # Assigning a List to a Name (line 83):
        
        # Assigning a List to a Name (line 83):
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_89253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        
        # Assigning a type to the variable 'nk' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'nk', list_89253)

        @norecursion
        def err_est(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'err_est'
            module_type_store = module_type_store.open_function_context('err_est', 85, 8, False)
            
            # Passed parameters checking function
            err_est.stypy_localization = localization
            err_est.stypy_type_of_self = None
            err_est.stypy_type_store = module_type_store
            err_est.stypy_function_name = 'err_est'
            err_est.stypy_param_names_list = ['k', 'd']
            err_est.stypy_varargs_param_name = None
            err_est.stypy_kwargs_param_name = None
            err_est.stypy_call_defaults = defaults
            err_est.stypy_call_varargs = varargs
            err_est.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'err_est', ['k', 'd'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'err_est', localization, ['k', 'd'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'err_est(...)' code ##################

            
            # Assigning a BinOp to a Name (line 87):
            
            # Assigning a BinOp to a Name (line 87):
            float_89254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'float')
            
            # Call to float(...): (line 87)
            # Processing the call arguments (line 87)
            # Getting the type of 'N' (line 87)
            N_89256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'N', False)
            # Processing the call keyword arguments (line 87)
            kwargs_89257 = {}
            # Getting the type of 'float' (line 87)
            float_89255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'float', False)
            # Calling float(args, kwargs) (line 87)
            float_call_result_89258 = invoke(stypy.reporting.localization.Localization(__file__, 87, 20), float_89255, *[N_89256], **kwargs_89257)
            
            # Applying the binary operator 'div' (line 87)
            result_div_89259 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 16), 'div', float_89254, float_call_result_89258)
            
            # Assigning a type to the variable 'h' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'h', result_div_89259)
            
            # Assigning a BinOp to a Name (line 88):
            
            # Assigning a BinOp to a Name (line 88):
            int_89260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'int')
            # Getting the type of 'h' (line 88)
            h_89261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'h')
            float_89262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'float')
            # Getting the type of 'k' (line 88)
            k_89263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'k')
            # Getting the type of 'd' (line 88)
            d_89264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'd')
            # Applying the binary operator '-' (line 88)
            result_sub_89265 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 31), '-', k_89263, d_89264)
            
            # Applying the binary operator '*' (line 88)
            result_mul_89266 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 26), '*', float_89262, result_sub_89265)
            
            # Applying the binary operator '**' (line 88)
            result_pow_89267 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 22), '**', h_89261, result_mul_89266)
            
            # Applying the binary operator '*' (line 88)
            result_mul_89268 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 18), '*', int_89260, result_pow_89267)
            
            # Assigning a type to the variable 'tol' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'tol', result_mul_89268)
            
            
            # Getting the type of 's' (line 89)
            s_89269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 's')
            int_89270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'int')
            # Applying the binary operator '>' (line 89)
            result_gt_89271 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), '>', s_89269, int_89270)
            
            # Testing the type of an if condition (line 89)
            if_condition_89272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_gt_89271)
            # Assigning a type to the variable 'if_condition_89272' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_89272', if_condition_89272)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'tol' (line 90)
            tol_89273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'tol')
            float_89274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
            # Getting the type of 's' (line 90)
            s_89275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 27), 's')
            # Applying the binary operator '*' (line 90)
            result_mul_89276 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '*', float_89274, s_89275)
            
            # Applying the binary operator '+=' (line 90)
            result_iadd_89277 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 16), '+=', tol_89273, result_mul_89276)
            # Assigning a type to the variable 'tol' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'tol', result_iadd_89277)
            
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'tol' (line 91)
            tol_89278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'tol')
            # Assigning a type to the variable 'stypy_return_type' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'stypy_return_type', tol_89278)
            
            # ################# End of 'err_est(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'err_est' in the type store
            # Getting the type of 'stypy_return_type' (line 85)
            stypy_return_type_89279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_89279)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'err_est'
            return stypy_return_type_89279

        # Assigning a type to the variable 'err_est' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'err_est', err_est)
        
        
        # Call to range(...): (line 93)
        # Processing the call arguments (line 93)
        int_89281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 23), 'int')
        int_89282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_89283 = {}
        # Getting the type of 'range' (line 93)
        range_89280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'range', False)
        # Calling range(args, kwargs) (line 93)
        range_call_result_89284 = invoke(stypy.reporting.localization.Localization(__file__, 93, 17), range_89280, *[int_89281, int_89282], **kwargs_89283)
        
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), range_call_result_89284)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_89285 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), range_call_result_89284)
        # Assigning a type to the variable 'k' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'k', for_loop_var_89285)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to splrep(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'x' (line 94)
        x_89287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'x', False)
        # Getting the type of 'v' (line 94)
        v_89288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 27), 'v', False)
        # Processing the call keyword arguments (line 94)
        # Getting the type of 's' (line 94)
        s_89289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 's', False)
        keyword_89290 = s_89289
        # Getting the type of 'per' (line 94)
        per_89291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'per', False)
        keyword_89292 = per_89291
        # Getting the type of 'k' (line 94)
        k_89293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'k', False)
        keyword_89294 = k_89293
        # Getting the type of 'xe' (line 94)
        xe_89295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 48), 'xe', False)
        keyword_89296 = xe_89295
        kwargs_89297 = {'s': keyword_89290, 'k': keyword_89294, 'per': keyword_89292, 'xe': keyword_89296}
        # Getting the type of 'splrep' (line 94)
        splrep_89286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'splrep', False)
        # Calling splrep(args, kwargs) (line 94)
        splrep_call_result_89298 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), splrep_89286, *[x_89287, v_89288], **kwargs_89297)
        
        # Assigning a type to the variable 'tck' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tck', splrep_call_result_89298)
        
        # Getting the type of 'at' (line 95)
        at_89299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'at')
        # Testing the type of an if condition (line 95)
        if_condition_89300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), at_89299)
        # Assigning a type to the variable 'if_condition_89300' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_89300', if_condition_89300)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 96):
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 96)
        k_89301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'k')
        
        # Getting the type of 'k' (line 96)
        k_89302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'k')
        # Applying the 'usub' unary operator (line 96)
        result___neg___89303 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 29), 'usub', k_89302)
        
        slice_89304 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 20), k_89301, result___neg___89303, None)
        
        # Obtaining the type of the subscript
        int_89305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'int')
        # Getting the type of 'tck' (line 96)
        tck_89306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'tck')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___89307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), tck_89306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_89308 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), getitem___89307, int_89305)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___89309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), subscript_call_result_89308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_89310 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), getitem___89309, slice_89304)
        
        # Assigning a type to the variable 't' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 't', subscript_call_result_89310)
        # SSA branch for the else part of an if statement (line 95)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 98):
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'x1' (line 98)
        x1_89311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'x1')
        # Assigning a type to the variable 't' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 't', x1_89311)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 99):
        
        # Assigning a List to a Name (line 99):
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_89312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        
        # Assigning a type to the variable 'nd' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'nd', list_89312)
        
        
        # Call to range(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'k' (line 100)
        k_89314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'k', False)
        int_89315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'int')
        # Applying the binary operator '+' (line 100)
        result_add_89316 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 27), '+', k_89314, int_89315)
        
        # Processing the call keyword arguments (line 100)
        kwargs_89317 = {}
        # Getting the type of 'range' (line 100)
        range_89313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'range', False)
        # Calling range(args, kwargs) (line 100)
        range_call_result_89318 = invoke(stypy.reporting.localization.Localization(__file__, 100, 21), range_89313, *[result_add_89316], **kwargs_89317)
        
        # Testing the type of a for loop iterable (line 100)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_89318)
        # Getting the type of the for loop variable (line 100)
        for_loop_var_89319 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_89318)
        # Assigning a type to the variable 'd' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'd', for_loop_var_89319)
        # SSA begins for a for statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to err_est(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'k' (line 101)
        k_89321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'k', False)
        # Getting the type of 'd' (line 101)
        d_89322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'd', False)
        # Processing the call keyword arguments (line 101)
        kwargs_89323 = {}
        # Getting the type of 'err_est' (line 101)
        err_est_89320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'err_est', False)
        # Calling err_est(args, kwargs) (line 101)
        err_est_call_result_89324 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), err_est_89320, *[k_89321, d_89322], **kwargs_89323)
        
        # Assigning a type to the variable 'tol' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'tol', err_est_call_result_89324)
        
        # Assigning a BinOp to a Name (line 102):
        
        # Assigning a BinOp to a Name (line 102):
        
        # Call to norm2(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to f(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 't' (line 102)
        t_89327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 't', False)
        # Getting the type of 'd' (line 102)
        d_89328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'd', False)
        # Processing the call keyword arguments (line 102)
        kwargs_89329 = {}
        # Getting the type of 'f' (line 102)
        f_89326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'f', False)
        # Calling f(args, kwargs) (line 102)
        f_call_result_89330 = invoke(stypy.reporting.localization.Localization(__file__, 102, 28), f_89326, *[t_89327, d_89328], **kwargs_89329)
        
        
        # Call to splev(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 't' (line 102)
        t_89332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 't', False)
        # Getting the type of 'tck' (line 102)
        tck_89333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'tck', False)
        # Getting the type of 'd' (line 102)
        d_89334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 47), 'd', False)
        # Processing the call keyword arguments (line 102)
        kwargs_89335 = {}
        # Getting the type of 'splev' (line 102)
        splev_89331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'splev', False)
        # Calling splev(args, kwargs) (line 102)
        splev_call_result_89336 = invoke(stypy.reporting.localization.Localization(__file__, 102, 35), splev_89331, *[t_89332, tck_89333, d_89334], **kwargs_89335)
        
        # Applying the binary operator '-' (line 102)
        result_sub_89337 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 28), '-', f_call_result_89330, splev_call_result_89336)
        
        # Processing the call keyword arguments (line 102)
        kwargs_89338 = {}
        # Getting the type of 'norm2' (line 102)
        norm2_89325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'norm2', False)
        # Calling norm2(args, kwargs) (line 102)
        norm2_call_result_89339 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), norm2_89325, *[result_sub_89337], **kwargs_89338)
        
        
        # Call to norm2(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to f(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 't' (line 102)
        t_89342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 61), 't', False)
        # Getting the type of 'd' (line 102)
        d_89343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 63), 'd', False)
        # Processing the call keyword arguments (line 102)
        kwargs_89344 = {}
        # Getting the type of 'f' (line 102)
        f_89341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 59), 'f', False)
        # Calling f(args, kwargs) (line 102)
        f_call_result_89345 = invoke(stypy.reporting.localization.Localization(__file__, 102, 59), f_89341, *[t_89342, d_89343], **kwargs_89344)
        
        # Processing the call keyword arguments (line 102)
        kwargs_89346 = {}
        # Getting the type of 'norm2' (line 102)
        norm2_89340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'norm2', False)
        # Calling norm2(args, kwargs) (line 102)
        norm2_call_result_89347 = invoke(stypy.reporting.localization.Localization(__file__, 102, 53), norm2_89340, *[f_call_result_89345], **kwargs_89346)
        
        # Applying the binary operator 'div' (line 102)
        result_div_89348 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 22), 'div', norm2_call_result_89339, norm2_call_result_89347)
        
        # Assigning a type to the variable 'err' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'err', result_div_89348)
        
        # Call to assert_(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Getting the type of 'err' (line 103)
        err_89350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'err', False)
        # Getting the type of 'tol' (line 103)
        tol_89351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'tol', False)
        # Applying the binary operator '<' (line 103)
        result_lt_89352 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 24), '<', err_89350, tol_89351)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_89353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'k' (line 103)
        k_89354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_89353, k_89354)
        # Adding element type (line 103)
        # Getting the type of 'd' (line 103)
        d_89355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_89353, d_89355)
        # Adding element type (line 103)
        # Getting the type of 'err' (line 103)
        err_89356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'err', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_89353, err_89356)
        # Adding element type (line 103)
        # Getting the type of 'tol' (line 103)
        tol_89357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 47), 'tol', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_89353, tol_89357)
        
        # Processing the call keyword arguments (line 103)
        kwargs_89358 = {}
        # Getting the type of 'assert_' (line 103)
        assert__89349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 103)
        assert__call_result_89359 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), assert__89349, *[result_lt_89352, tuple_89353], **kwargs_89358)
        
        
        # Call to append(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_89362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        # Getting the type of 'err' (line 104)
        err_89363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'err', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), tuple_89362, err_89363)
        # Adding element type (line 104)
        # Getting the type of 'tol' (line 104)
        tol_89364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'tol', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 27), tuple_89362, tol_89364)
        
        # Processing the call keyword arguments (line 104)
        kwargs_89365 = {}
        # Getting the type of 'nd' (line 104)
        nd_89360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'nd', False)
        # Obtaining the member 'append' of a type (line 104)
        append_89361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), nd_89360, 'append')
        # Calling append(args, kwargs) (line 104)
        append_call_result_89366 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), append_89361, *[tuple_89362], **kwargs_89365)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'nd' (line 105)
        nd_89369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'nd', False)
        # Processing the call keyword arguments (line 105)
        kwargs_89370 = {}
        # Getting the type of 'nk' (line 105)
        nk_89367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'nk', False)
        # Obtaining the member 'append' of a type (line 105)
        append_89368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), nk_89367, 'append')
        # Calling append(args, kwargs) (line 105)
        append_call_result_89371 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), append_89368, *[nd_89369], **kwargs_89370)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 106)
        # Processing the call arguments (line 106)
        str_89373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 12), 'str', '\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_89374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        
        # Call to f(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'None' (line 106)
        None_89376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 68), 'None', False)
        # Processing the call keyword arguments (line 106)
        kwargs_89377 = {}
        # Getting the type of 'f' (line 106)
        f_89375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 66), 'f', False)
        # Calling f(args, kwargs) (line 106)
        f_call_result_89378 = invoke(stypy.reporting.localization.Localization(__file__, 106, 66), f_89375, *[None_89376], **kwargs_89377)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 66), tuple_89374, f_call_result_89378)
        # Adding element type (line 106)
        
        # Call to repr(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to round(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'xb' (line 107)
        xb_89381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 67), 'xb', False)
        int_89382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 70), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_89383 = {}
        # Getting the type of 'round' (line 107)
        round_89380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 61), 'round', False)
        # Calling round(args, kwargs) (line 107)
        round_call_result_89384 = invoke(stypy.reporting.localization.Localization(__file__, 107, 61), round_89380, *[xb_89381, int_89382], **kwargs_89383)
        
        # Processing the call keyword arguments (line 107)
        kwargs_89385 = {}
        # Getting the type of 'repr' (line 107)
        repr_89379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 56), 'repr', False)
        # Calling repr(args, kwargs) (line 107)
        repr_call_result_89386 = invoke(stypy.reporting.localization.Localization(__file__, 107, 56), repr_89379, *[round_call_result_89384], **kwargs_89385)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 66), tuple_89374, repr_call_result_89386)
        # Adding element type (line 106)
        
        # Call to repr(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Call to round(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'xe' (line 107)
        xe_89389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 85), 'xe', False)
        int_89390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 88), 'int')
        # Processing the call keyword arguments (line 107)
        kwargs_89391 = {}
        # Getting the type of 'round' (line 107)
        round_89388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 79), 'round', False)
        # Calling round(args, kwargs) (line 107)
        round_call_result_89392 = invoke(stypy.reporting.localization.Localization(__file__, 107, 79), round_89388, *[xe_89389, int_89390], **kwargs_89391)
        
        # Processing the call keyword arguments (line 107)
        kwargs_89393 = {}
        # Getting the type of 'repr' (line 107)
        repr_89387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 74), 'repr', False)
        # Calling repr(args, kwargs) (line 107)
        repr_call_result_89394 = invoke(stypy.reporting.localization.Localization(__file__, 107, 74), repr_89387, *[round_call_result_89392], **kwargs_89393)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 66), tuple_89374, repr_call_result_89394)
        # Adding element type (line 106)
        
        # Call to repr(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to round(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'a' (line 108)
        a_89397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 69), 'a', False)
        int_89398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 71), 'int')
        # Processing the call keyword arguments (line 108)
        kwargs_89399 = {}
        # Getting the type of 'round' (line 108)
        round_89396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 63), 'round', False)
        # Calling round(args, kwargs) (line 108)
        round_call_result_89400 = invoke(stypy.reporting.localization.Localization(__file__, 108, 63), round_89396, *[a_89397, int_89398], **kwargs_89399)
        
        # Processing the call keyword arguments (line 108)
        kwargs_89401 = {}
        # Getting the type of 'repr' (line 108)
        repr_89395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 58), 'repr', False)
        # Calling repr(args, kwargs) (line 108)
        repr_call_result_89402 = invoke(stypy.reporting.localization.Localization(__file__, 108, 58), repr_89395, *[round_call_result_89400], **kwargs_89401)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 66), tuple_89374, repr_call_result_89402)
        # Adding element type (line 106)
        
        # Call to repr(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to round(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'b' (line 108)
        b_89405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 86), 'b', False)
        int_89406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 88), 'int')
        # Processing the call keyword arguments (line 108)
        kwargs_89407 = {}
        # Getting the type of 'round' (line 108)
        round_89404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 80), 'round', False)
        # Calling round(args, kwargs) (line 108)
        round_call_result_89408 = invoke(stypy.reporting.localization.Localization(__file__, 108, 80), round_89404, *[b_89405, int_89406], **kwargs_89407)
        
        # Processing the call keyword arguments (line 108)
        kwargs_89409 = {}
        # Getting the type of 'repr' (line 108)
        repr_89403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 75), 'repr', False)
        # Calling repr(args, kwargs) (line 108)
        repr_call_result_89410 = invoke(stypy.reporting.localization.Localization(__file__, 108, 75), repr_89403, *[round_call_result_89408], **kwargs_89409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 66), tuple_89374, repr_call_result_89410)
        
        # Applying the binary operator '%' (line 106)
        result_mod_89411 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), '%', str_89373, tuple_89374)
        
        # Processing the call keyword arguments (line 106)
        kwargs_89412 = {}
        # Getting the type of 'put' (line 106)
        put_89372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'put', False)
        # Calling put(args, kwargs) (line 106)
        put_call_result_89413 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), put_89372, *[result_mod_89411], **kwargs_89412)
        
        
        # Getting the type of 'at' (line 109)
        at_89414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'at')
        # Testing the type of an if condition (line 109)
        if_condition_89415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), at_89414)
        # Assigning a type to the variable 'if_condition_89415' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_89415', if_condition_89415)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 110):
        
        # Assigning a Str to a Name (line 110):
        str_89416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'str', 'at knots')
        # Assigning a type to the variable 'str' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'str', str_89416)
        # SSA branch for the else part of an if statement (line 109)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 112):
        
        # Assigning a Str to a Name (line 112):
        str_89417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 18), 'str', 'at the middle of nodes')
        # Assigning a type to the variable 'str' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'str', str_89417)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 113)
        # Processing the call arguments (line 113)
        str_89419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'str', ' per=%d s=%s Evaluation %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_89420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'per' (line 113)
        per_89421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 44), 'per', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 44), tuple_89420, per_89421)
        # Adding element type (line 113)
        
        # Call to repr(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 's' (line 113)
        s_89423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 53), 's', False)
        # Processing the call keyword arguments (line 113)
        kwargs_89424 = {}
        # Getting the type of 'repr' (line 113)
        repr_89422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'repr', False)
        # Calling repr(args, kwargs) (line 113)
        repr_call_result_89425 = invoke(stypy.reporting.localization.Localization(__file__, 113, 48), repr_89422, *[s_89423], **kwargs_89424)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 44), tuple_89420, repr_call_result_89425)
        # Adding element type (line 113)
        # Getting the type of 'str' (line 113)
        str_89426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 56), 'str', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 44), tuple_89420, str_89426)
        
        # Applying the binary operator '%' (line 113)
        result_mod_89427 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 12), '%', str_89419, tuple_89420)
        
        # Processing the call keyword arguments (line 113)
        kwargs_89428 = {}
        # Getting the type of 'put' (line 113)
        put_89418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'put', False)
        # Calling put(args, kwargs) (line 113)
        put_call_result_89429 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), put_89418, *[result_mod_89427], **kwargs_89428)
        
        
        # Call to put(...): (line 114)
        # Processing the call arguments (line 114)
        str_89431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', " k :  |f-s|^2  |f'-s'| |f''-.. |f'''-. |f''''- |f'''''")
        # Processing the call keyword arguments (line 114)
        kwargs_89432 = {}
        # Getting the type of 'put' (line 114)
        put_89430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'put', False)
        # Calling put(args, kwargs) (line 114)
        put_call_result_89433 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), put_89430, *[str_89431], **kwargs_89432)
        
        
        # Assigning a Num to a Name (line 115):
        
        # Assigning a Num to a Name (line 115):
        int_89434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
        # Assigning a type to the variable 'k' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'k', int_89434)
        
        # Getting the type of 'nk' (line 116)
        nk_89435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'nk')
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 8), nk_89435)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_89436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 8), nk_89435)
        # Assigning a type to the variable 'l' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'l', for_loop_var_89436)
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to put(...): (line 117)
        # Processing the call arguments (line 117)
        str_89438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'str', ' %d : ')
        # Getting the type of 'k' (line 117)
        k_89439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'k', False)
        # Applying the binary operator '%' (line 117)
        result_mod_89440 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), '%', str_89438, k_89439)
        
        # Processing the call keyword arguments (line 117)
        kwargs_89441 = {}
        # Getting the type of 'put' (line 117)
        put_89437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'put', False)
        # Calling put(args, kwargs) (line 117)
        put_call_result_89442 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), put_89437, *[result_mod_89440], **kwargs_89441)
        
        
        # Getting the type of 'l' (line 118)
        l_89443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'l')
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), l_89443)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_89444 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), l_89443)
        # Assigning a type to the variable 'r' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'r', for_loop_var_89444)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to put(...): (line 119)
        # Processing the call arguments (line 119)
        str_89446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'str', ' %.1e  %.1e')
        # Getting the type of 'r' (line 119)
        r_89447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'r', False)
        # Applying the binary operator '%' (line 119)
        result_mod_89448 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), '%', str_89446, r_89447)
        
        # Processing the call keyword arguments (line 119)
        kwargs_89449 = {}
        # Getting the type of 'put' (line 119)
        put_89445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'put', False)
        # Calling put(args, kwargs) (line 119)
        put_call_result_89450 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), put_89445, *[result_mod_89448], **kwargs_89449)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 120)
        # Processing the call arguments (line 120)
        str_89452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 16), 'str', '\n')
        # Processing the call keyword arguments (line 120)
        kwargs_89453 = {}
        # Getting the type of 'put' (line 120)
        put_89451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'put', False)
        # Calling put(args, kwargs) (line 120)
        put_call_result_89454 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), put_89451, *[str_89452], **kwargs_89453)
        
        
        # Assigning a BinOp to a Name (line 121):
        
        # Assigning a BinOp to a Name (line 121):
        # Getting the type of 'k' (line 121)
        k_89455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'k')
        int_89456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'int')
        # Applying the binary operator '+' (line 121)
        result_add_89457 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '+', k_89455, int_89456)
        
        # Assigning a type to the variable 'k' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'k', result_add_89457)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_1' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_89458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_89458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_1'
        return stypy_return_type_89458


    @norecursion
    def check_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'f1' (line 123)
        f1_89459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'f1')
        int_89460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'int')
        int_89461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'int')
        int_89462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 38), 'int')
        int_89463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 42), 'int')
        # Getting the type of 'pi' (line 123)
        pi_89464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'pi')
        # Applying the binary operator '*' (line 123)
        result_mul_89465 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 42), '*', int_89463, pi_89464)
        
        int_89466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        # Getting the type of 'None' (line 123)
        None_89467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 55), 'None')
        # Getting the type of 'None' (line 123)
        None_89468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'None')
        int_89469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'int')
        int_89470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'int')
        # Getting the type of 'pi' (line 124)
        pi_89471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'pi')
        # Applying the binary operator '*' (line 124)
        result_mul_89472 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 22), '*', int_89470, pi_89471)
        
        float_89473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'float')
        # Getting the type of 'pi' (line 124)
        pi_89474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'pi')
        # Applying the binary operator '*' (line 124)
        result_mul_89475 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 30), '*', float_89473, pi_89474)
        
        defaults = [f1_89459, int_89460, int_89461, int_89462, result_mul_89465, int_89466, None_89467, None_89468, int_89469, result_mul_89472, result_mul_89475]
        # Create a new context for function 'check_2'
        module_type_store = module_type_store.open_function_context('check_2', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.check_2')
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_param_names_list', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'])
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.check_2.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.check_2', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_2', localization, ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_2(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 125)
        # Getting the type of 'xb' (line 125)
        xb_89476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'xb')
        # Getting the type of 'None' (line 125)
        None_89477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'None')
        
        (may_be_89478, more_types_in_union_89479) = may_be_none(xb_89476, None_89477)

        if may_be_89478:

            if more_types_in_union_89479:
                # Runtime conditional SSA (line 125)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 126):
            
            # Assigning a Name to a Name (line 126):
            # Getting the type of 'a' (line 126)
            a_89480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'a')
            # Assigning a type to the variable 'xb' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'xb', a_89480)

            if more_types_in_union_89479:
                # SSA join for if statement (line 125)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 127)
        # Getting the type of 'xe' (line 127)
        xe_89481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'xe')
        # Getting the type of 'None' (line 127)
        None_89482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'None')
        
        (may_be_89483, more_types_in_union_89484) = may_be_none(xe_89481, None_89482)

        if may_be_89483:

            if more_types_in_union_89484:
                # Runtime conditional SSA (line 127)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 128):
            
            # Assigning a Name to a Name (line 128):
            # Getting the type of 'b' (line 128)
            b_89485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'b')
            # Assigning a type to the variable 'xe' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'xe', b_89485)

            if more_types_in_union_89484:
                # SSA join for if statement (line 127)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        # Getting the type of 'a' (line 129)
        a_89486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'a')
        # Getting the type of 'b' (line 129)
        b_89487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'b')
        # Getting the type of 'a' (line 129)
        a_89488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'a')
        # Applying the binary operator '-' (line 129)
        result_sub_89489 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '-', b_89487, a_89488)
        
        
        # Call to arange(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'N' (line 129)
        N_89491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'N', False)
        int_89492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'int')
        # Applying the binary operator '+' (line 129)
        result_add_89493 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 27), '+', N_89491, int_89492)
        
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'float' (line 129)
        float_89494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'float', False)
        keyword_89495 = float_89494
        kwargs_89496 = {'dtype': keyword_89495}
        # Getting the type of 'arange' (line 129)
        arange_89490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 129)
        arange_call_result_89497 = invoke(stypy.reporting.localization.Localization(__file__, 129, 20), arange_89490, *[result_add_89493], **kwargs_89496)
        
        # Applying the binary operator '*' (line 129)
        result_mul_89498 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 14), '*', result_sub_89489, arange_call_result_89497)
        
        
        # Call to float(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'N' (line 129)
        N_89500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'N', False)
        # Processing the call keyword arguments (line 129)
        kwargs_89501 = {}
        # Getting the type of 'float' (line 129)
        float_89499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'float', False)
        # Calling float(args, kwargs) (line 129)
        float_call_result_89502 = invoke(stypy.reporting.localization.Localization(__file__, 129, 44), float_89499, *[N_89500], **kwargs_89501)
        
        # Applying the binary operator 'div' (line 129)
        result_div_89503 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 43), 'div', result_mul_89498, float_call_result_89502)
        
        # Applying the binary operator '+' (line 129)
        result_add_89504 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 12), '+', a_89486, result_div_89503)
        
        # Assigning a type to the variable 'x' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'x', result_add_89504)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to f(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'x' (line 130)
        x_89506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), 'x', False)
        # Processing the call keyword arguments (line 130)
        kwargs_89507 = {}
        # Getting the type of 'f' (line 130)
        f_89505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'f', False)
        # Calling f(args, kwargs) (line 130)
        f_call_result_89508 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), f_89505, *[x_89506], **kwargs_89507)
        
        # Assigning a type to the variable 'v' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'v', f_call_result_89508)

        @norecursion
        def err_est(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'err_est'
            module_type_store = module_type_store.open_function_context('err_est', 132, 8, False)
            
            # Passed parameters checking function
            err_est.stypy_localization = localization
            err_est.stypy_type_of_self = None
            err_est.stypy_type_store = module_type_store
            err_est.stypy_function_name = 'err_est'
            err_est.stypy_param_names_list = ['k', 'd']
            err_est.stypy_varargs_param_name = None
            err_est.stypy_kwargs_param_name = None
            err_est.stypy_call_defaults = defaults
            err_est.stypy_call_varargs = varargs
            err_est.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'err_est', ['k', 'd'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'err_est', localization, ['k', 'd'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'err_est(...)' code ##################

            
            # Assigning a BinOp to a Name (line 134):
            
            # Assigning a BinOp to a Name (line 134):
            float_89509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 16), 'float')
            
            # Call to float(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 'N' (line 134)
            N_89511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'N', False)
            # Processing the call keyword arguments (line 134)
            kwargs_89512 = {}
            # Getting the type of 'float' (line 134)
            float_89510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'float', False)
            # Calling float(args, kwargs) (line 134)
            float_call_result_89513 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), float_89510, *[N_89511], **kwargs_89512)
            
            # Applying the binary operator 'div' (line 134)
            result_div_89514 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), 'div', float_89509, float_call_result_89513)
            
            # Assigning a type to the variable 'h' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'h', result_div_89514)
            
            # Assigning a BinOp to a Name (line 135):
            
            # Assigning a BinOp to a Name (line 135):
            int_89515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'int')
            # Getting the type of 'h' (line 135)
            h_89516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'h')
            float_89517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 26), 'float')
            # Getting the type of 'k' (line 135)
            k_89518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'k')
            # Getting the type of 'd' (line 135)
            d_89519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'd')
            # Applying the binary operator '-' (line 135)
            result_sub_89520 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 31), '-', k_89518, d_89519)
            
            # Applying the binary operator '*' (line 135)
            result_mul_89521 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 26), '*', float_89517, result_sub_89520)
            
            # Applying the binary operator '**' (line 135)
            result_pow_89522 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 22), '**', h_89516, result_mul_89521)
            
            # Applying the binary operator '*' (line 135)
            result_mul_89523 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 18), '*', int_89515, result_pow_89522)
            
            # Assigning a type to the variable 'tol' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'tol', result_mul_89523)
            
            
            # Getting the type of 's' (line 136)
            s_89524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 's')
            int_89525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'int')
            # Applying the binary operator '>' (line 136)
            result_gt_89526 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 15), '>', s_89524, int_89525)
            
            # Testing the type of an if condition (line 136)
            if_condition_89527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 12), result_gt_89526)
            # Assigning a type to the variable 'if_condition_89527' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'if_condition_89527', if_condition_89527)
            # SSA begins for if statement (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'tol' (line 137)
            tol_89528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tol')
            float_89529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'float')
            # Getting the type of 's' (line 137)
            s_89530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 's')
            # Applying the binary operator '*' (line 137)
            result_mul_89531 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 23), '*', float_89529, s_89530)
            
            # Applying the binary operator '+=' (line 137)
            result_iadd_89532 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), '+=', tol_89528, result_mul_89531)
            # Assigning a type to the variable 'tol' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'tol', result_iadd_89532)
            
            # SSA join for if statement (line 136)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'tol' (line 138)
            tol_89533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'tol')
            # Assigning a type to the variable 'stypy_return_type' (line 138)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', tol_89533)
            
            # ################# End of 'err_est(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'err_est' in the type store
            # Getting the type of 'stypy_return_type' (line 132)
            stypy_return_type_89534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_89534)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'err_est'
            return stypy_return_type_89534

        # Assigning a type to the variable 'err_est' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'err_est', err_est)
        
        # Assigning a List to a Name (line 140):
        
        # Assigning a List to a Name (line 140):
        
        # Obtaining an instance of the builtin type 'list' (line 140)
        list_89535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 140)
        
        # Assigning a type to the variable 'nk' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'nk', list_89535)
        
        
        # Call to range(...): (line 141)
        # Processing the call arguments (line 141)
        int_89537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 23), 'int')
        int_89538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'int')
        # Processing the call keyword arguments (line 141)
        kwargs_89539 = {}
        # Getting the type of 'range' (line 141)
        range_89536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'range', False)
        # Calling range(args, kwargs) (line 141)
        range_call_result_89540 = invoke(stypy.reporting.localization.Localization(__file__, 141, 17), range_89536, *[int_89537, int_89538], **kwargs_89539)
        
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), range_call_result_89540)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_89541 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), range_call_result_89540)
        # Assigning a type to the variable 'k' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'k', for_loop_var_89541)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to splrep(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_89543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 25), 'x', False)
        # Getting the type of 'v' (line 142)
        v_89544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'v', False)
        # Processing the call keyword arguments (line 142)
        # Getting the type of 's' (line 142)
        s_89545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 's', False)
        keyword_89546 = s_89545
        # Getting the type of 'per' (line 142)
        per_89547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'per', False)
        keyword_89548 = per_89547
        # Getting the type of 'k' (line 142)
        k_89549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'k', False)
        keyword_89550 = k_89549
        # Getting the type of 'xe' (line 142)
        xe_89551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'xe', False)
        keyword_89552 = xe_89551
        kwargs_89553 = {'s': keyword_89546, 'k': keyword_89550, 'per': keyword_89548, 'xe': keyword_89552}
        # Getting the type of 'splrep' (line 142)
        splrep_89542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'splrep', False)
        # Calling splrep(args, kwargs) (line 142)
        splrep_call_result_89554 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), splrep_89542, *[x_89543, v_89544], **kwargs_89553)
        
        # Assigning a type to the variable 'tck' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'tck', splrep_call_result_89554)
        
        # Call to append(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_89557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        
        # Call to splint(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'ia' (line 143)
        ia_89559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'ia', False)
        # Getting the type of 'ib' (line 143)
        ib_89560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'ib', False)
        # Getting the type of 'tck' (line 143)
        tck_89561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 36), 'tck', False)
        # Processing the call keyword arguments (line 143)
        kwargs_89562 = {}
        # Getting the type of 'splint' (line 143)
        splint_89558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'splint', False)
        # Calling splint(args, kwargs) (line 143)
        splint_call_result_89563 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), splint_89558, *[ia_89559, ib_89560, tck_89561], **kwargs_89562)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), list_89557, splint_call_result_89563)
        # Adding element type (line 143)
        
        # Call to spalde(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'dx' (line 143)
        dx_89565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 48), 'dx', False)
        # Getting the type of 'tck' (line 143)
        tck_89566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 51), 'tck', False)
        # Processing the call keyword arguments (line 143)
        kwargs_89567 = {}
        # Getting the type of 'spalde' (line 143)
        spalde_89564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 41), 'spalde', False)
        # Calling spalde(args, kwargs) (line 143)
        spalde_call_result_89568 = invoke(stypy.reporting.localization.Localization(__file__, 143, 41), spalde_89564, *[dx_89565, tck_89566], **kwargs_89567)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), list_89557, spalde_call_result_89568)
        
        # Processing the call keyword arguments (line 143)
        kwargs_89569 = {}
        # Getting the type of 'nk' (line 143)
        nk_89555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'nk', False)
        # Obtaining the member 'append' of a type (line 143)
        append_89556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), nk_89555, 'append')
        # Calling append(args, kwargs) (line 143)
        append_call_result_89570 = invoke(stypy.reporting.localization.Localization(__file__, 143, 12), append_89556, *[list_89557], **kwargs_89569)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 144)
        # Processing the call arguments (line 144)
        str_89572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 12), 'str', '\nf = %s  s=S_k(x;t,c)  x in [%s, %s] > [%s, %s]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 144)
        tuple_89573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 144)
        # Adding element type (line 144)
        
        # Call to f(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'None' (line 144)
        None_89575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 68), 'None', False)
        # Processing the call keyword arguments (line 144)
        kwargs_89576 = {}
        # Getting the type of 'f' (line 144)
        f_89574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 66), 'f', False)
        # Calling f(args, kwargs) (line 144)
        f_call_result_89577 = invoke(stypy.reporting.localization.Localization(__file__, 144, 66), f_89574, *[None_89575], **kwargs_89576)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 66), tuple_89573, f_call_result_89577)
        # Adding element type (line 144)
        
        # Call to repr(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to round(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'xb' (line 145)
        xb_89580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 62), 'xb', False)
        int_89581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 65), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_89582 = {}
        # Getting the type of 'round' (line 145)
        round_89579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 56), 'round', False)
        # Calling round(args, kwargs) (line 145)
        round_call_result_89583 = invoke(stypy.reporting.localization.Localization(__file__, 145, 56), round_89579, *[xb_89580, int_89581], **kwargs_89582)
        
        # Processing the call keyword arguments (line 145)
        kwargs_89584 = {}
        # Getting the type of 'repr' (line 145)
        repr_89578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 51), 'repr', False)
        # Calling repr(args, kwargs) (line 145)
        repr_call_result_89585 = invoke(stypy.reporting.localization.Localization(__file__, 145, 51), repr_89578, *[round_call_result_89583], **kwargs_89584)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 66), tuple_89573, repr_call_result_89585)
        # Adding element type (line 144)
        
        # Call to repr(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to round(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'xe' (line 145)
        xe_89588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 80), 'xe', False)
        int_89589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 83), 'int')
        # Processing the call keyword arguments (line 145)
        kwargs_89590 = {}
        # Getting the type of 'round' (line 145)
        round_89587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 74), 'round', False)
        # Calling round(args, kwargs) (line 145)
        round_call_result_89591 = invoke(stypy.reporting.localization.Localization(__file__, 145, 74), round_89587, *[xe_89588, int_89589], **kwargs_89590)
        
        # Processing the call keyword arguments (line 145)
        kwargs_89592 = {}
        # Getting the type of 'repr' (line 145)
        repr_89586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 69), 'repr', False)
        # Calling repr(args, kwargs) (line 145)
        repr_call_result_89593 = invoke(stypy.reporting.localization.Localization(__file__, 145, 69), repr_89586, *[round_call_result_89591], **kwargs_89592)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 66), tuple_89573, repr_call_result_89593)
        # Adding element type (line 144)
        
        # Call to repr(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to round(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'a' (line 146)
        a_89596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 63), 'a', False)
        int_89597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 65), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_89598 = {}
        # Getting the type of 'round' (line 146)
        round_89595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 57), 'round', False)
        # Calling round(args, kwargs) (line 146)
        round_call_result_89599 = invoke(stypy.reporting.localization.Localization(__file__, 146, 57), round_89595, *[a_89596, int_89597], **kwargs_89598)
        
        # Processing the call keyword arguments (line 146)
        kwargs_89600 = {}
        # Getting the type of 'repr' (line 146)
        repr_89594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 52), 'repr', False)
        # Calling repr(args, kwargs) (line 146)
        repr_call_result_89601 = invoke(stypy.reporting.localization.Localization(__file__, 146, 52), repr_89594, *[round_call_result_89599], **kwargs_89600)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 66), tuple_89573, repr_call_result_89601)
        # Adding element type (line 144)
        
        # Call to repr(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to round(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'b' (line 146)
        b_89604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 80), 'b', False)
        int_89605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 82), 'int')
        # Processing the call keyword arguments (line 146)
        kwargs_89606 = {}
        # Getting the type of 'round' (line 146)
        round_89603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 74), 'round', False)
        # Calling round(args, kwargs) (line 146)
        round_call_result_89607 = invoke(stypy.reporting.localization.Localization(__file__, 146, 74), round_89603, *[b_89604, int_89605], **kwargs_89606)
        
        # Processing the call keyword arguments (line 146)
        kwargs_89608 = {}
        # Getting the type of 'repr' (line 146)
        repr_89602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 69), 'repr', False)
        # Calling repr(args, kwargs) (line 146)
        repr_call_result_89609 = invoke(stypy.reporting.localization.Localization(__file__, 146, 69), repr_89602, *[round_call_result_89607], **kwargs_89608)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 66), tuple_89573, repr_call_result_89609)
        
        # Applying the binary operator '%' (line 144)
        result_mod_89610 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '%', str_89572, tuple_89573)
        
        # Processing the call keyword arguments (line 144)
        kwargs_89611 = {}
        # Getting the type of 'put' (line 144)
        put_89571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'put', False)
        # Calling put(args, kwargs) (line 144)
        put_call_result_89612 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), put_89571, *[result_mod_89610], **kwargs_89611)
        
        
        # Call to put(...): (line 147)
        # Processing the call arguments (line 147)
        str_89614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'str', ' per=%d s=%s N=%d [a, b] = [%s, %s]  dx=%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_89615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 60), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        # Getting the type of 'per' (line 147)
        per_89616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 60), 'per', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, per_89616)
        # Adding element type (line 147)
        
        # Call to repr(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 's' (line 147)
        s_89618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 69), 's', False)
        # Processing the call keyword arguments (line 147)
        kwargs_89619 = {}
        # Getting the type of 'repr' (line 147)
        repr_89617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 64), 'repr', False)
        # Calling repr(args, kwargs) (line 147)
        repr_call_result_89620 = invoke(stypy.reporting.localization.Localization(__file__, 147, 64), repr_89617, *[s_89618], **kwargs_89619)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, repr_call_result_89620)
        # Adding element type (line 147)
        # Getting the type of 'N' (line 147)
        N_89621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 72), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, N_89621)
        # Adding element type (line 147)
        
        # Call to repr(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to round(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'ia' (line 147)
        ia_89624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 85), 'ia', False)
        int_89625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 88), 'int')
        # Processing the call keyword arguments (line 147)
        kwargs_89626 = {}
        # Getting the type of 'round' (line 147)
        round_89623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 79), 'round', False)
        # Calling round(args, kwargs) (line 147)
        round_call_result_89627 = invoke(stypy.reporting.localization.Localization(__file__, 147, 79), round_89623, *[ia_89624, int_89625], **kwargs_89626)
        
        # Processing the call keyword arguments (line 147)
        kwargs_89628 = {}
        # Getting the type of 'repr' (line 147)
        repr_89622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 74), 'repr', False)
        # Calling repr(args, kwargs) (line 147)
        repr_call_result_89629 = invoke(stypy.reporting.localization.Localization(__file__, 147, 74), repr_89622, *[round_call_result_89627], **kwargs_89628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, repr_call_result_89629)
        # Adding element type (line 147)
        
        # Call to repr(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to round(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'ib' (line 147)
        ib_89632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 103), 'ib', False)
        int_89633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 106), 'int')
        # Processing the call keyword arguments (line 147)
        kwargs_89634 = {}
        # Getting the type of 'round' (line 147)
        round_89631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 97), 'round', False)
        # Calling round(args, kwargs) (line 147)
        round_call_result_89635 = invoke(stypy.reporting.localization.Localization(__file__, 147, 97), round_89631, *[ib_89632, int_89633], **kwargs_89634)
        
        # Processing the call keyword arguments (line 147)
        kwargs_89636 = {}
        # Getting the type of 'repr' (line 147)
        repr_89630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 92), 'repr', False)
        # Calling repr(args, kwargs) (line 147)
        repr_call_result_89637 = invoke(stypy.reporting.localization.Localization(__file__, 147, 92), repr_89630, *[round_call_result_89635], **kwargs_89636)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, repr_call_result_89637)
        # Adding element type (line 147)
        
        # Call to repr(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to round(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'dx' (line 147)
        dx_89640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 121), 'dx', False)
        int_89641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 124), 'int')
        # Processing the call keyword arguments (line 147)
        kwargs_89642 = {}
        # Getting the type of 'round' (line 147)
        round_89639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 115), 'round', False)
        # Calling round(args, kwargs) (line 147)
        round_call_result_89643 = invoke(stypy.reporting.localization.Localization(__file__, 147, 115), round_89639, *[dx_89640, int_89641], **kwargs_89642)
        
        # Processing the call keyword arguments (line 147)
        kwargs_89644 = {}
        # Getting the type of 'repr' (line 147)
        repr_89638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 110), 'repr', False)
        # Calling repr(args, kwargs) (line 147)
        repr_call_result_89645 = invoke(stypy.reporting.localization.Localization(__file__, 147, 110), repr_89638, *[round_call_result_89643], **kwargs_89644)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 60), tuple_89615, repr_call_result_89645)
        
        # Applying the binary operator '%' (line 147)
        result_mod_89646 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 12), '%', str_89614, tuple_89615)
        
        # Processing the call keyword arguments (line 147)
        kwargs_89647 = {}
        # Getting the type of 'put' (line 147)
        put_89613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'put', False)
        # Calling put(args, kwargs) (line 147)
        put_call_result_89648 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), put_89613, *[result_mod_89646], **kwargs_89647)
        
        
        # Call to put(...): (line 148)
        # Processing the call arguments (line 148)
        str_89650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 12), 'str', ' k :  int(s,[a,b]) Int.Error   Rel. error of s^(d)(dx) d = 0, .., k')
        # Processing the call keyword arguments (line 148)
        kwargs_89651 = {}
        # Getting the type of 'put' (line 148)
        put_89649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'put', False)
        # Calling put(args, kwargs) (line 148)
        put_call_result_89652 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), put_89649, *[str_89650], **kwargs_89651)
        
        
        # Assigning a Num to a Name (line 149):
        
        # Assigning a Num to a Name (line 149):
        int_89653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
        # Assigning a type to the variable 'k' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'k', int_89653)
        
        # Getting the type of 'nk' (line 150)
        nk_89654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'nk')
        # Testing the type of a for loop iterable (line 150)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 150, 8), nk_89654)
        # Getting the type of the for loop variable (line 150)
        for_loop_var_89655 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 150, 8), nk_89654)
        # Assigning a type to the variable 'r' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'r', for_loop_var_89655)
        # SSA begins for a for statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_89656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 17), 'int')
        # Getting the type of 'r' (line 151)
        r_89657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'r')
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___89658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), r_89657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_89659 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), getitem___89658, int_89656)
        
        int_89660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'int')
        # Applying the binary operator '<' (line 151)
        result_lt_89661 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 15), '<', subscript_call_result_89659, int_89660)
        
        # Testing the type of an if condition (line 151)
        if_condition_89662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 12), result_lt_89661)
        # Assigning a type to the variable 'if_condition_89662' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'if_condition_89662', if_condition_89662)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 152):
        
        # Assigning a Str to a Name (line 152):
        str_89663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'str', '-')
        # Assigning a type to the variable 'sr' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'sr', str_89663)
        # SSA branch for the else part of an if statement (line 151)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 154):
        
        # Assigning a Str to a Name (line 154):
        str_89664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'str', ' ')
        # Assigning a type to the variable 'sr' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'sr', str_89664)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 155)
        # Processing the call arguments (line 155)
        str_89666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'str', ' %d   %s%.8f   %.1e ')
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_89667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'k' (line 155)
        k_89668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 42), tuple_89667, k_89668)
        # Adding element type (line 155)
        # Getting the type of 'sr' (line 155)
        sr_89669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 44), 'sr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 42), tuple_89667, sr_89669)
        # Adding element type (line 155)
        
        # Call to abs(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Obtaining the type of the subscript
        int_89671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 53), 'int')
        # Getting the type of 'r' (line 155)
        r_89672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___89673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 51), r_89672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_89674 = invoke(stypy.reporting.localization.Localization(__file__, 155, 51), getitem___89673, int_89671)
        
        # Processing the call keyword arguments (line 155)
        kwargs_89675 = {}
        # Getting the type of 'abs' (line 155)
        abs_89670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 47), 'abs', False)
        # Calling abs(args, kwargs) (line 155)
        abs_call_result_89676 = invoke(stypy.reporting.localization.Localization(__file__, 155, 47), abs_89670, *[subscript_call_result_89674], **kwargs_89675)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 42), tuple_89667, abs_call_result_89676)
        # Adding element type (line 155)
        
        # Call to abs(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining the type of the subscript
        int_89678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 47), 'int')
        # Getting the type of 'r' (line 156)
        r_89679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'r', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___89680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 45), r_89679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_89681 = invoke(stypy.reporting.localization.Localization(__file__, 156, 45), getitem___89680, int_89678)
        
        
        # Call to f(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'ib' (line 156)
        ib_89683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 53), 'ib', False)
        int_89684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 56), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_89685 = {}
        # Getting the type of 'f' (line 156)
        f_89682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 51), 'f', False)
        # Calling f(args, kwargs) (line 156)
        f_call_result_89686 = invoke(stypy.reporting.localization.Localization(__file__, 156, 51), f_89682, *[ib_89683, int_89684], **kwargs_89685)
        
        
        # Call to f(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'ia' (line 156)
        ia_89688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 62), 'ia', False)
        int_89689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 65), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_89690 = {}
        # Getting the type of 'f' (line 156)
        f_89687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'f', False)
        # Calling f(args, kwargs) (line 156)
        f_call_result_89691 = invoke(stypy.reporting.localization.Localization(__file__, 156, 60), f_89687, *[ia_89688, int_89689], **kwargs_89690)
        
        # Applying the binary operator '-' (line 156)
        result_sub_89692 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 51), '-', f_call_result_89686, f_call_result_89691)
        
        # Applying the binary operator '-' (line 156)
        result_sub_89693 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 45), '-', subscript_call_result_89681, result_sub_89692)
        
        # Processing the call keyword arguments (line 156)
        kwargs_89694 = {}
        # Getting the type of 'abs' (line 156)
        abs_89677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'abs', False)
        # Calling abs(args, kwargs) (line 156)
        abs_call_result_89695 = invoke(stypy.reporting.localization.Localization(__file__, 156, 41), abs_89677, *[result_sub_89693], **kwargs_89694)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 42), tuple_89667, abs_call_result_89695)
        
        # Applying the binary operator '%' (line 155)
        result_mod_89696 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 16), '%', str_89666, tuple_89667)
        
        # Processing the call keyword arguments (line 155)
        kwargs_89697 = {}
        # Getting the type of 'put' (line 155)
        put_89665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'put', False)
        # Calling put(args, kwargs) (line 155)
        put_call_result_89698 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), put_89665, *[result_mod_89696], **kwargs_89697)
        
        
        # Assigning a Num to a Name (line 157):
        
        # Assigning a Num to a Name (line 157):
        int_89699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 16), 'int')
        # Assigning a type to the variable 'd' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'd', int_89699)
        
        
        # Obtaining the type of the subscript
        int_89700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'int')
        # Getting the type of 'r' (line 158)
        r_89701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'r')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___89702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), r_89701, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_89703 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), getitem___89702, int_89700)
        
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 12), subscript_call_result_89703)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_89704 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 12), subscript_call_result_89703)
        # Assigning a type to the variable 'dr' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'dr', for_loop_var_89704)
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to abs(...): (line 159)
        # Processing the call arguments (line 159)
        int_89706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'int')
        # Getting the type of 'dr' (line 159)
        dr_89707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 28), 'dr', False)
        
        # Call to f(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'dx' (line 159)
        dx_89709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'dx', False)
        # Getting the type of 'd' (line 159)
        d_89710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'd', False)
        # Processing the call keyword arguments (line 159)
        kwargs_89711 = {}
        # Getting the type of 'f' (line 159)
        f_89708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'f', False)
        # Calling f(args, kwargs) (line 159)
        f_call_result_89712 = invoke(stypy.reporting.localization.Localization(__file__, 159, 31), f_89708, *[dx_89709, d_89710], **kwargs_89711)
        
        # Applying the binary operator 'div' (line 159)
        result_div_89713 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 28), 'div', dr_89707, f_call_result_89712)
        
        # Applying the binary operator '-' (line 159)
        result_sub_89714 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 26), '-', int_89706, result_div_89713)
        
        # Processing the call keyword arguments (line 159)
        kwargs_89715 = {}
        # Getting the type of 'abs' (line 159)
        abs_89705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'abs', False)
        # Calling abs(args, kwargs) (line 159)
        abs_call_result_89716 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), abs_89705, *[result_sub_89714], **kwargs_89715)
        
        # Assigning a type to the variable 'err' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'err', abs_call_result_89716)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to err_est(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'k' (line 160)
        k_89718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'k', False)
        # Getting the type of 'd' (line 160)
        d_89719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'd', False)
        # Processing the call keyword arguments (line 160)
        kwargs_89720 = {}
        # Getting the type of 'err_est' (line 160)
        err_est_89717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'err_est', False)
        # Calling err_est(args, kwargs) (line 160)
        err_est_call_result_89721 = invoke(stypy.reporting.localization.Localization(__file__, 160, 22), err_est_89717, *[k_89718, d_89719], **kwargs_89720)
        
        # Assigning a type to the variable 'tol' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'tol', err_est_call_result_89721)
        
        # Call to assert_(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Getting the type of 'err' (line 161)
        err_89723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'err', False)
        # Getting the type of 'tol' (line 161)
        tol_89724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 30), 'tol', False)
        # Applying the binary operator '<' (line 161)
        result_lt_89725 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), '<', err_89723, tol_89724)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_89726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'k' (line 161)
        k_89727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), tuple_89726, k_89727)
        # Adding element type (line 161)
        # Getting the type of 'd' (line 161)
        d_89728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 39), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 36), tuple_89726, d_89728)
        
        # Processing the call keyword arguments (line 161)
        kwargs_89729 = {}
        # Getting the type of 'assert_' (line 161)
        assert__89722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 161)
        assert__call_result_89730 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), assert__89722, *[result_lt_89725, tuple_89726], **kwargs_89729)
        
        
        # Call to put(...): (line 162)
        # Processing the call arguments (line 162)
        str_89732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'str', ' %.1e %.1e')
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_89733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        # Getting the type of 'err' (line 162)
        err_89734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 36), 'err', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), tuple_89733, err_89734)
        # Adding element type (line 162)
        # Getting the type of 'tol' (line 162)
        tol_89735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'tol', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 36), tuple_89733, tol_89735)
        
        # Applying the binary operator '%' (line 162)
        result_mod_89736 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 20), '%', str_89732, tuple_89733)
        
        # Processing the call keyword arguments (line 162)
        kwargs_89737 = {}
        # Getting the type of 'put' (line 162)
        put_89731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'put', False)
        # Calling put(args, kwargs) (line 162)
        put_call_result_89738 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), put_89731, *[result_mod_89736], **kwargs_89737)
        
        
        # Assigning a BinOp to a Name (line 163):
        
        # Assigning a BinOp to a Name (line 163):
        # Getting the type of 'd' (line 163)
        d_89739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'd')
        int_89740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 22), 'int')
        # Applying the binary operator '+' (line 163)
        result_add_89741 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '+', d_89739, int_89740)
        
        # Assigning a type to the variable 'd' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'd', result_add_89741)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 164)
        # Processing the call arguments (line 164)
        str_89743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 16), 'str', '\n')
        # Processing the call keyword arguments (line 164)
        kwargs_89744 = {}
        # Getting the type of 'put' (line 164)
        put_89742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'put', False)
        # Calling put(args, kwargs) (line 164)
        put_call_result_89745 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), put_89742, *[str_89743], **kwargs_89744)
        
        
        # Assigning a BinOp to a Name (line 165):
        
        # Assigning a BinOp to a Name (line 165):
        # Getting the type of 'k' (line 165)
        k_89746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'k')
        int_89747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'int')
        # Applying the binary operator '+' (line 165)
        result_add_89748 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '+', k_89746, int_89747)
        
        # Assigning a type to the variable 'k' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'k', result_add_89748)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_2' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_89749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_89749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_2'
        return stypy_return_type_89749


    @norecursion
    def check_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'f1' (line 167)
        f1_89750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 23), 'f1')
        int_89751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 30), 'int')
        int_89752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 34), 'int')
        int_89753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 38), 'int')
        int_89754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 42), 'int')
        # Getting the type of 'pi' (line 167)
        pi_89755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'pi')
        # Applying the binary operator '*' (line 167)
        result_mul_89756 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 42), '*', int_89754, pi_89755)
        
        int_89757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 49), 'int')
        # Getting the type of 'None' (line 167)
        None_89758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 55), 'None')
        # Getting the type of 'None' (line 167)
        None_89759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 63), 'None')
        int_89760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 17), 'int')
        int_89761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 22), 'int')
        # Getting the type of 'pi' (line 168)
        pi_89762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'pi')
        # Applying the binary operator '*' (line 168)
        result_mul_89763 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 22), '*', int_89761, pi_89762)
        
        float_89764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 30), 'float')
        # Getting the type of 'pi' (line 168)
        pi_89765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'pi')
        # Applying the binary operator '*' (line 168)
        result_mul_89766 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 30), '*', float_89764, pi_89765)
        
        defaults = [f1_89750, int_89751, int_89752, int_89753, result_mul_89756, int_89757, None_89758, None_89759, int_89760, result_mul_89763, result_mul_89766]
        # Create a new context for function 'check_3'
        module_type_store = module_type_store.open_function_context('check_3', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.check_3')
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_param_names_list', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'])
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.check_3.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.check_3', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_3', localization, ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_3(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 169)
        # Getting the type of 'xb' (line 169)
        xb_89767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'xb')
        # Getting the type of 'None' (line 169)
        None_89768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'None')
        
        (may_be_89769, more_types_in_union_89770) = may_be_none(xb_89767, None_89768)

        if may_be_89769:

            if more_types_in_union_89770:
                # Runtime conditional SSA (line 169)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 170):
            
            # Assigning a Name to a Name (line 170):
            # Getting the type of 'a' (line 170)
            a_89771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 17), 'a')
            # Assigning a type to the variable 'xb' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'xb', a_89771)

            if more_types_in_union_89770:
                # SSA join for if statement (line 169)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 171)
        # Getting the type of 'xe' (line 171)
        xe_89772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'xe')
        # Getting the type of 'None' (line 171)
        None_89773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'None')
        
        (may_be_89774, more_types_in_union_89775) = may_be_none(xe_89772, None_89773)

        if may_be_89774:

            if more_types_in_union_89775:
                # Runtime conditional SSA (line 171)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 172):
            
            # Assigning a Name to a Name (line 172):
            # Getting the type of 'b' (line 172)
            b_89776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 17), 'b')
            # Assigning a type to the variable 'xe' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'xe', b_89776)

            if more_types_in_union_89775:
                # SSA join for if statement (line 171)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 173):
        
        # Assigning a BinOp to a Name (line 173):
        # Getting the type of 'a' (line 173)
        a_89777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'a')
        # Getting the type of 'b' (line 173)
        b_89778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'b')
        # Getting the type of 'a' (line 173)
        a_89779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'a')
        # Applying the binary operator '-' (line 173)
        result_sub_89780 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '-', b_89778, a_89779)
        
        
        # Call to arange(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'N' (line 173)
        N_89782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 27), 'N', False)
        int_89783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 29), 'int')
        # Applying the binary operator '+' (line 173)
        result_add_89784 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 27), '+', N_89782, int_89783)
        
        # Processing the call keyword arguments (line 173)
        # Getting the type of 'float' (line 173)
        float_89785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), 'float', False)
        keyword_89786 = float_89785
        kwargs_89787 = {'dtype': keyword_89786}
        # Getting the type of 'arange' (line 173)
        arange_89781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 173)
        arange_call_result_89788 = invoke(stypy.reporting.localization.Localization(__file__, 173, 20), arange_89781, *[result_add_89784], **kwargs_89787)
        
        # Applying the binary operator '*' (line 173)
        result_mul_89789 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 14), '*', result_sub_89780, arange_call_result_89788)
        
        
        # Call to float(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'N' (line 173)
        N_89791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'N', False)
        # Processing the call keyword arguments (line 173)
        kwargs_89792 = {}
        # Getting the type of 'float' (line 173)
        float_89790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 44), 'float', False)
        # Calling float(args, kwargs) (line 173)
        float_call_result_89793 = invoke(stypy.reporting.localization.Localization(__file__, 173, 44), float_89790, *[N_89791], **kwargs_89792)
        
        # Applying the binary operator 'div' (line 173)
        result_div_89794 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 43), 'div', result_mul_89789, float_call_result_89793)
        
        # Applying the binary operator '+' (line 173)
        result_add_89795 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 12), '+', a_89777, result_div_89794)
        
        # Assigning a type to the variable 'x' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'x', result_add_89795)
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to f(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'x' (line 174)
        x_89797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'x', False)
        # Processing the call keyword arguments (line 174)
        kwargs_89798 = {}
        # Getting the type of 'f' (line 174)
        f_89796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'f', False)
        # Calling f(args, kwargs) (line 174)
        f_call_result_89799 = invoke(stypy.reporting.localization.Localization(__file__, 174, 12), f_89796, *[x_89797], **kwargs_89798)
        
        # Assigning a type to the variable 'v' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'v', f_call_result_89799)
        
        # Call to put(...): (line 175)
        # Processing the call arguments (line 175)
        str_89801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 12), 'str', '  k  :     Roots of s(x) approx %s  x in [%s,%s]:')
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_89802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        
        # Call to f(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'None' (line 176)
        None_89804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'None', False)
        # Processing the call keyword arguments (line 176)
        kwargs_89805 = {}
        # Getting the type of 'f' (line 176)
        f_89803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'f', False)
        # Calling f(args, kwargs) (line 176)
        f_call_result_89806 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), f_89803, *[None_89804], **kwargs_89805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 15), tuple_89802, f_call_result_89806)
        # Adding element type (line 176)
        
        # Call to repr(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to round(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'a' (line 176)
        a_89809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'a', False)
        int_89810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 36), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_89811 = {}
        # Getting the type of 'round' (line 176)
        round_89808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'round', False)
        # Calling round(args, kwargs) (line 176)
        round_call_result_89812 = invoke(stypy.reporting.localization.Localization(__file__, 176, 28), round_89808, *[a_89809, int_89810], **kwargs_89811)
        
        # Processing the call keyword arguments (line 176)
        kwargs_89813 = {}
        # Getting the type of 'repr' (line 176)
        repr_89807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'repr', False)
        # Calling repr(args, kwargs) (line 176)
        repr_call_result_89814 = invoke(stypy.reporting.localization.Localization(__file__, 176, 23), repr_89807, *[round_call_result_89812], **kwargs_89813)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 15), tuple_89802, repr_call_result_89814)
        # Adding element type (line 176)
        
        # Call to repr(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to round(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'b' (line 176)
        b_89817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 51), 'b', False)
        int_89818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 53), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_89819 = {}
        # Getting the type of 'round' (line 176)
        round_89816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'round', False)
        # Calling round(args, kwargs) (line 176)
        round_call_result_89820 = invoke(stypy.reporting.localization.Localization(__file__, 176, 45), round_89816, *[b_89817, int_89818], **kwargs_89819)
        
        # Processing the call keyword arguments (line 176)
        kwargs_89821 = {}
        # Getting the type of 'repr' (line 176)
        repr_89815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 40), 'repr', False)
        # Calling repr(args, kwargs) (line 176)
        repr_call_result_89822 = invoke(stypy.reporting.localization.Localization(__file__, 176, 40), repr_89815, *[round_call_result_89820], **kwargs_89821)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 15), tuple_89802, repr_call_result_89822)
        
        # Applying the binary operator '%' (line 175)
        result_mod_89823 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 12), '%', str_89801, tuple_89802)
        
        # Processing the call keyword arguments (line 175)
        kwargs_89824 = {}
        # Getting the type of 'put' (line 175)
        put_89800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'put', False)
        # Calling put(args, kwargs) (line 175)
        put_call_result_89825 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), put_89800, *[result_mod_89823], **kwargs_89824)
        
        
        
        # Call to range(...): (line 177)
        # Processing the call arguments (line 177)
        int_89827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'int')
        int_89828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'int')
        # Processing the call keyword arguments (line 177)
        kwargs_89829 = {}
        # Getting the type of 'range' (line 177)
        range_89826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'range', False)
        # Calling range(args, kwargs) (line 177)
        range_call_result_89830 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), range_89826, *[int_89827, int_89828], **kwargs_89829)
        
        # Testing the type of a for loop iterable (line 177)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_89830)
        # Getting the type of the for loop variable (line 177)
        for_loop_var_89831 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 177, 8), range_call_result_89830)
        # Assigning a type to the variable 'k' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'k', for_loop_var_89831)
        # SSA begins for a for statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 178):
        
        # Assigning a Call to a Name (line 178):
        
        # Call to splrep(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'x' (line 178)
        x_89833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'x', False)
        # Getting the type of 'v' (line 178)
        v_89834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'v', False)
        # Processing the call keyword arguments (line 178)
        # Getting the type of 's' (line 178)
        s_89835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 's', False)
        keyword_89836 = s_89835
        # Getting the type of 'per' (line 178)
        per_89837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'per', False)
        keyword_89838 = per_89837
        # Getting the type of 'k' (line 178)
        k_89839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 47), 'k', False)
        keyword_89840 = k_89839
        # Getting the type of 'xe' (line 178)
        xe_89841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 53), 'xe', False)
        keyword_89842 = xe_89841
        kwargs_89843 = {'s': keyword_89836, 'k': keyword_89840, 'per': keyword_89838, 'xe': keyword_89842}
        # Getting the type of 'splrep' (line 178)
        splrep_89832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'splrep', False)
        # Calling splrep(args, kwargs) (line 178)
        splrep_call_result_89844 = invoke(stypy.reporting.localization.Localization(__file__, 178, 18), splrep_89832, *[x_89833, v_89834], **kwargs_89843)
        
        # Assigning a type to the variable 'tck' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'tck', splrep_call_result_89844)
        
        
        # Getting the type of 'k' (line 179)
        k_89845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'k')
        int_89846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'int')
        # Applying the binary operator '==' (line 179)
        result_eq_89847 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 15), '==', k_89845, int_89846)
        
        # Testing the type of an if condition (line 179)
        if_condition_89848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 12), result_eq_89847)
        # Assigning a type to the variable 'if_condition_89848' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'if_condition_89848', if_condition_89848)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 180):
        
        # Assigning a Call to a Name (line 180):
        
        # Call to sproot(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'tck' (line 180)
        tck_89850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'tck', False)
        # Processing the call keyword arguments (line 180)
        kwargs_89851 = {}
        # Getting the type of 'sproot' (line 180)
        sproot_89849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'sproot', False)
        # Calling sproot(args, kwargs) (line 180)
        sproot_call_result_89852 = invoke(stypy.reporting.localization.Localization(__file__, 180, 24), sproot_89849, *[tck_89850], **kwargs_89851)
        
        # Assigning a type to the variable 'roots' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'roots', sproot_call_result_89852)
        
        # Call to assert_allclose(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to splev(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'roots' (line 181)
        roots_89855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 38), 'roots', False)
        # Getting the type of 'tck' (line 181)
        tck_89856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 45), 'tck', False)
        # Processing the call keyword arguments (line 181)
        kwargs_89857 = {}
        # Getting the type of 'splev' (line 181)
        splev_89854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'splev', False)
        # Calling splev(args, kwargs) (line 181)
        splev_call_result_89858 = invoke(stypy.reporting.localization.Localization(__file__, 181, 32), splev_89854, *[roots_89855, tck_89856], **kwargs_89857)
        
        int_89859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 51), 'int')
        # Processing the call keyword arguments (line 181)
        float_89860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 59), 'float')
        keyword_89861 = float_89860
        float_89862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 71), 'float')
        keyword_89863 = float_89862
        kwargs_89864 = {'rtol': keyword_89863, 'atol': keyword_89861}
        # Getting the type of 'assert_allclose' (line 181)
        assert_allclose_89853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 181)
        assert_allclose_call_result_89865 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), assert_allclose_89853, *[splev_call_result_89858, int_89859], **kwargs_89864)
        
        
        # Call to assert_allclose(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'roots' (line 182)
        roots_89867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 32), 'roots', False)
        # Getting the type of 'pi' (line 182)
        pi_89868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'pi', False)
        
        # Call to array(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_89870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        int_89871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 48), list_89870, int_89871)
        # Adding element type (line 182)
        int_89872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 48), list_89870, int_89872)
        # Adding element type (line 182)
        int_89873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 48), list_89870, int_89873)
        # Adding element type (line 182)
        int_89874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 48), list_89870, int_89874)
        
        # Processing the call keyword arguments (line 182)
        kwargs_89875 = {}
        # Getting the type of 'array' (line 182)
        array_89869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'array', False)
        # Calling array(args, kwargs) (line 182)
        array_call_result_89876 = invoke(stypy.reporting.localization.Localization(__file__, 182, 42), array_89869, *[list_89870], **kwargs_89875)
        
        # Applying the binary operator '*' (line 182)
        result_mul_89877 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 39), '*', pi_89868, array_call_result_89876)
        
        # Processing the call keyword arguments (line 182)
        float_89878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 68), 'float')
        keyword_89879 = float_89878
        kwargs_89880 = {'rtol': keyword_89879}
        # Getting the type of 'assert_allclose' (line 182)
        assert_allclose_89866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 182)
        assert_allclose_call_result_89881 = invoke(stypy.reporting.localization.Localization(__file__, 182, 16), assert_allclose_89866, *[roots_89867, result_mul_89877], **kwargs_89880)
        
        
        # Call to put(...): (line 183)
        # Processing the call arguments (line 183)
        str_89883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'str', '  %d  : %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_89884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'k' (line 183)
        k_89885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 36), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 36), tuple_89884, k_89885)
        # Adding element type (line 183)
        
        # Call to repr(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Call to tolist(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_89889 = {}
        # Getting the type of 'roots' (line 183)
        roots_89887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 44), 'roots', False)
        # Obtaining the member 'tolist' of a type (line 183)
        tolist_89888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 44), roots_89887, 'tolist')
        # Calling tolist(args, kwargs) (line 183)
        tolist_call_result_89890 = invoke(stypy.reporting.localization.Localization(__file__, 183, 44), tolist_89888, *[], **kwargs_89889)
        
        # Processing the call keyword arguments (line 183)
        kwargs_89891 = {}
        # Getting the type of 'repr' (line 183)
        repr_89886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'repr', False)
        # Calling repr(args, kwargs) (line 183)
        repr_call_result_89892 = invoke(stypy.reporting.localization.Localization(__file__, 183, 39), repr_89886, *[tolist_call_result_89890], **kwargs_89891)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 36), tuple_89884, repr_call_result_89892)
        
        # Applying the binary operator '%' (line 183)
        result_mod_89893 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 20), '%', str_89883, tuple_89884)
        
        # Processing the call keyword arguments (line 183)
        kwargs_89894 = {}
        # Getting the type of 'put' (line 183)
        put_89882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'put', False)
        # Calling put(args, kwargs) (line 183)
        put_call_result_89895 = invoke(stypy.reporting.localization.Localization(__file__, 183, 16), put_89882, *[result_mod_89893], **kwargs_89894)
        
        # SSA branch for the else part of an if statement (line 179)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_raises(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'ValueError' (line 185)
        ValueError_89897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'ValueError', False)
        # Getting the type of 'sproot' (line 185)
        sproot_89898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'sproot', False)
        # Getting the type of 'tck' (line 185)
        tck_89899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 50), 'tck', False)
        # Processing the call keyword arguments (line 185)
        kwargs_89900 = {}
        # Getting the type of 'assert_raises' (line 185)
        assert_raises_89896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 185)
        assert_raises_call_result_89901 = invoke(stypy.reporting.localization.Localization(__file__, 185, 16), assert_raises_89896, *[ValueError_89897, sproot_89898, tck_89899], **kwargs_89900)
        
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_3' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_89902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_89902)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_3'
        return stypy_return_type_89902


    @norecursion
    def check_4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'f1' (line 187)
        f1_89903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'f1')
        int_89904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'int')
        int_89905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 34), 'int')
        int_89906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'int')
        int_89907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 42), 'int')
        # Getting the type of 'pi' (line 187)
        pi_89908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 'pi')
        # Applying the binary operator '*' (line 187)
        result_mul_89909 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 42), '*', int_89907, pi_89908)
        
        int_89910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 49), 'int')
        # Getting the type of 'None' (line 187)
        None_89911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 55), 'None')
        # Getting the type of 'None' (line 187)
        None_89912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 63), 'None')
        int_89913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 17), 'int')
        int_89914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 22), 'int')
        # Getting the type of 'pi' (line 188)
        pi_89915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'pi')
        # Applying the binary operator '*' (line 188)
        result_mul_89916 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 22), '*', int_89914, pi_89915)
        
        float_89917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 30), 'float')
        # Getting the type of 'pi' (line 188)
        pi_89918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 34), 'pi')
        # Applying the binary operator '*' (line 188)
        result_mul_89919 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 30), '*', float_89917, pi_89918)
        
        defaults = [f1_89903, int_89904, int_89905, int_89906, result_mul_89909, int_89910, None_89911, None_89912, int_89913, result_mul_89916, result_mul_89919]
        # Create a new context for function 'check_4'
        module_type_store = module_type_store.open_function_context('check_4', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.check_4')
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_param_names_list', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'])
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.check_4.__dict__.__setitem__('stypy_declared_arg_number', 12)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.check_4', ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_4', localization, ['f', 'per', 's', 'a', 'b', 'N', 'xb', 'xe', 'ia', 'ib', 'dx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_4(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 189)
        # Getting the type of 'xb' (line 189)
        xb_89920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'xb')
        # Getting the type of 'None' (line 189)
        None_89921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 17), 'None')
        
        (may_be_89922, more_types_in_union_89923) = may_be_none(xb_89920, None_89921)

        if may_be_89922:

            if more_types_in_union_89923:
                # Runtime conditional SSA (line 189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 190):
            
            # Assigning a Name to a Name (line 190):
            # Getting the type of 'a' (line 190)
            a_89924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 17), 'a')
            # Assigning a type to the variable 'xb' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'xb', a_89924)

            if more_types_in_union_89923:
                # SSA join for if statement (line 189)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 191)
        # Getting the type of 'xe' (line 191)
        xe_89925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'xe')
        # Getting the type of 'None' (line 191)
        None_89926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 17), 'None')
        
        (may_be_89927, more_types_in_union_89928) = may_be_none(xe_89925, None_89926)

        if may_be_89927:

            if more_types_in_union_89928:
                # Runtime conditional SSA (line 191)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 192):
            
            # Assigning a Name to a Name (line 192):
            # Getting the type of 'b' (line 192)
            b_89929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'b')
            # Assigning a type to the variable 'xe' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'xe', b_89929)

            if more_types_in_union_89928:
                # SSA join for if statement (line 191)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 193):
        
        # Assigning a BinOp to a Name (line 193):
        # Getting the type of 'a' (line 193)
        a_89930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'a')
        # Getting the type of 'b' (line 193)
        b_89931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'b')
        # Getting the type of 'a' (line 193)
        a_89932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'a')
        # Applying the binary operator '-' (line 193)
        result_sub_89933 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 15), '-', b_89931, a_89932)
        
        
        # Call to arange(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'N' (line 193)
        N_89935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'N', False)
        int_89936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'int')
        # Applying the binary operator '+' (line 193)
        result_add_89937 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 27), '+', N_89935, int_89936)
        
        # Processing the call keyword arguments (line 193)
        # Getting the type of 'float' (line 193)
        float_89938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'float', False)
        keyword_89939 = float_89938
        kwargs_89940 = {'dtype': keyword_89939}
        # Getting the type of 'arange' (line 193)
        arange_89934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 193)
        arange_call_result_89941 = invoke(stypy.reporting.localization.Localization(__file__, 193, 20), arange_89934, *[result_add_89937], **kwargs_89940)
        
        # Applying the binary operator '*' (line 193)
        result_mul_89942 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 14), '*', result_sub_89933, arange_call_result_89941)
        
        
        # Call to float(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'N' (line 193)
        N_89944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 50), 'N', False)
        # Processing the call keyword arguments (line 193)
        kwargs_89945 = {}
        # Getting the type of 'float' (line 193)
        float_89943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 44), 'float', False)
        # Calling float(args, kwargs) (line 193)
        float_call_result_89946 = invoke(stypy.reporting.localization.Localization(__file__, 193, 44), float_89943, *[N_89944], **kwargs_89945)
        
        # Applying the binary operator 'div' (line 193)
        result_div_89947 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 43), 'div', result_mul_89942, float_call_result_89946)
        
        # Applying the binary operator '+' (line 193)
        result_add_89948 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '+', a_89930, result_div_89947)
        
        # Assigning a type to the variable 'x' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'x', result_add_89948)
        
        # Assigning a BinOp to a Name (line 194):
        
        # Assigning a BinOp to a Name (line 194):
        # Getting the type of 'a' (line 194)
        a_89949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 13), 'a')
        # Getting the type of 'b' (line 194)
        b_89950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'b')
        # Getting the type of 'a' (line 194)
        a_89951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'a')
        # Applying the binary operator '-' (line 194)
        result_sub_89952 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 18), '-', b_89950, a_89951)
        
        
        # Call to arange(...): (line 194)
        # Processing the call arguments (line 194)
        int_89954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 30), 'int')
        # Getting the type of 'N' (line 194)
        N_89955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 32), 'N', False)
        # Processing the call keyword arguments (line 194)
        # Getting the type of 'float' (line 194)
        float_89956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 40), 'float', False)
        keyword_89957 = float_89956
        kwargs_89958 = {'dtype': keyword_89957}
        # Getting the type of 'arange' (line 194)
        arange_89953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'arange', False)
        # Calling arange(args, kwargs) (line 194)
        arange_call_result_89959 = invoke(stypy.reporting.localization.Localization(__file__, 194, 23), arange_89953, *[int_89954, N_89955], **kwargs_89958)
        
        # Applying the binary operator '*' (line 194)
        result_mul_89960 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 17), '*', result_sub_89952, arange_call_result_89959)
        
        
        # Call to float(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'N' (line 194)
        N_89962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 53), 'N', False)
        int_89963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 55), 'int')
        # Applying the binary operator '-' (line 194)
        result_sub_89964 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 53), '-', N_89962, int_89963)
        
        # Processing the call keyword arguments (line 194)
        kwargs_89965 = {}
        # Getting the type of 'float' (line 194)
        float_89961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 47), 'float', False)
        # Calling float(args, kwargs) (line 194)
        float_call_result_89966 = invoke(stypy.reporting.localization.Localization(__file__, 194, 47), float_89961, *[result_sub_89964], **kwargs_89965)
        
        # Applying the binary operator 'div' (line 194)
        result_div_89967 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 46), 'div', result_mul_89960, float_call_result_89966)
        
        # Applying the binary operator '+' (line 194)
        result_add_89968 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 13), '+', a_89949, result_div_89967)
        
        # Assigning a type to the variable 'x1' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'x1', result_add_89968)
        
        # Assigning a Tuple to a Tuple (line 195):
        
        # Assigning a Call to a Name (line 195):
        
        # Call to f(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'x' (line 195)
        x_89970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'x', False)
        # Processing the call keyword arguments (line 195)
        kwargs_89971 = {}
        # Getting the type of 'f' (line 195)
        f_89969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'f', False)
        # Calling f(args, kwargs) (line 195)
        f_call_result_89972 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), f_89969, *[x_89970], **kwargs_89971)
        
        # Assigning a type to the variable 'tuple_assignment_88950' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'tuple_assignment_88950', f_call_result_89972)
        
        # Assigning a Call to a Name (line 195):
        
        # Call to f(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'x1' (line 195)
        x1_89974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'x1', False)
        # Processing the call keyword arguments (line 195)
        kwargs_89975 = {}
        # Getting the type of 'f' (line 195)
        f_89973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'f', False)
        # Calling f(args, kwargs) (line 195)
        f_call_result_89976 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), f_89973, *[x1_89974], **kwargs_89975)
        
        # Assigning a type to the variable 'tuple_assignment_88951' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'tuple_assignment_88951', f_call_result_89976)
        
        # Assigning a Name to a Name (line 195):
        # Getting the type of 'tuple_assignment_88950' (line 195)
        tuple_assignment_88950_89977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'tuple_assignment_88950')
        # Assigning a type to the variable 'v' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'v', tuple_assignment_88950_89977)
        
        # Assigning a Name to a Name (line 195):
        # Getting the type of 'tuple_assignment_88951' (line 195)
        tuple_assignment_88951_89978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'tuple_assignment_88951')
        # Assigning a type to the variable 'v1' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 10), 'v1', tuple_assignment_88951_89978)
        
        # Call to put(...): (line 196)
        # Processing the call arguments (line 196)
        str_89980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'str', ' u = %s   N = %d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 196)
        tuple_89981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 196)
        # Adding element type (line 196)
        
        # Call to repr(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to round(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'dx' (line 196)
        dx_89984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 45), 'dx', False)
        int_89985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 48), 'int')
        # Processing the call keyword arguments (line 196)
        kwargs_89986 = {}
        # Getting the type of 'round' (line 196)
        round_89983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'round', False)
        # Calling round(args, kwargs) (line 196)
        round_call_result_89987 = invoke(stypy.reporting.localization.Localization(__file__, 196, 39), round_89983, *[dx_89984, int_89985], **kwargs_89986)
        
        # Processing the call keyword arguments (line 196)
        kwargs_89988 = {}
        # Getting the type of 'repr' (line 196)
        repr_89982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), 'repr', False)
        # Calling repr(args, kwargs) (line 196)
        repr_call_result_89989 = invoke(stypy.reporting.localization.Localization(__file__, 196, 34), repr_89982, *[round_call_result_89987], **kwargs_89988)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 34), tuple_89981, repr_call_result_89989)
        # Adding element type (line 196)
        # Getting the type of 'N' (line 196)
        N_89990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 52), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 34), tuple_89981, N_89990)
        
        # Applying the binary operator '%' (line 196)
        result_mod_89991 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 12), '%', str_89980, tuple_89981)
        
        # Processing the call keyword arguments (line 196)
        kwargs_89992 = {}
        # Getting the type of 'put' (line 196)
        put_89979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'put', False)
        # Calling put(args, kwargs) (line 196)
        put_call_result_89993 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), put_89979, *[result_mod_89991], **kwargs_89992)
        
        
        # Call to put(...): (line 197)
        # Processing the call arguments (line 197)
        str_89995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 12), 'str', '  k  :  [x(u), %s(x(u))]  Error of splprep  Error of splrep ')
        
        # Call to f(...): (line 197)
        # Processing the call arguments (line 197)
        int_89997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 80), 'int')
        # Getting the type of 'None' (line 197)
        None_89998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 82), 'None', False)
        # Processing the call keyword arguments (line 197)
        kwargs_89999 = {}
        # Getting the type of 'f' (line 197)
        f_89996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 78), 'f', False)
        # Calling f(args, kwargs) (line 197)
        f_call_result_90000 = invoke(stypy.reporting.localization.Localization(__file__, 197, 78), f_89996, *[int_89997, None_89998], **kwargs_89999)
        
        # Applying the binary operator '%' (line 197)
        result_mod_90001 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 12), '%', str_89995, f_call_result_90000)
        
        # Processing the call keyword arguments (line 197)
        kwargs_90002 = {}
        # Getting the type of 'put' (line 197)
        put_89994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'put', False)
        # Calling put(args, kwargs) (line 197)
        put_call_result_90003 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), put_89994, *[result_mod_90001], **kwargs_90002)
        
        
        
        # Call to range(...): (line 198)
        # Processing the call arguments (line 198)
        int_90005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
        int_90006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 25), 'int')
        # Processing the call keyword arguments (line 198)
        kwargs_90007 = {}
        # Getting the type of 'range' (line 198)
        range_90004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'range', False)
        # Calling range(args, kwargs) (line 198)
        range_call_result_90008 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), range_90004, *[int_90005, int_90006], **kwargs_90007)
        
        # Testing the type of a for loop iterable (line 198)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 8), range_call_result_90008)
        # Getting the type of the for loop variable (line 198)
        for_loop_var_90009 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 8), range_call_result_90008)
        # Assigning a type to the variable 'k' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'k', for_loop_var_90009)
        # SSA begins for a for statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 199):
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_90010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 12), 'int')
        
        # Call to splprep(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_90012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        # Getting the type of 'x' (line 199)
        x_90013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 29), list_90012, x_90013)
        # Adding element type (line 199)
        # Getting the type of 'v' (line 199)
        v_90014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 29), list_90012, v_90014)
        
        # Processing the call keyword arguments (line 199)
        # Getting the type of 's' (line 199)
        s_90015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 's', False)
        keyword_90016 = s_90015
        # Getting the type of 'per' (line 199)
        per_90017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'per', False)
        keyword_90018 = per_90017
        # Getting the type of 'k' (line 199)
        k_90019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 49), 'k', False)
        keyword_90020 = k_90019
        int_90021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 56), 'int')
        keyword_90022 = int_90021
        kwargs_90023 = {'nest': keyword_90022, 's': keyword_90016, 'k': keyword_90020, 'per': keyword_90018}
        # Getting the type of 'splprep' (line 199)
        splprep_90011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'splprep', False)
        # Calling splprep(args, kwargs) (line 199)
        splprep_call_result_90024 = invoke(stypy.reporting.localization.Localization(__file__, 199, 21), splprep_90011, *[list_90012], **kwargs_90023)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___90025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), splprep_call_result_90024, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_90026 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), getitem___90025, int_90010)
        
        # Assigning a type to the variable 'tuple_var_assignment_88952' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'tuple_var_assignment_88952', subscript_call_result_90026)
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_90027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 12), 'int')
        
        # Call to splprep(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_90029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        # Getting the type of 'x' (line 199)
        x_90030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 29), list_90029, x_90030)
        # Adding element type (line 199)
        # Getting the type of 'v' (line 199)
        v_90031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 29), list_90029, v_90031)
        
        # Processing the call keyword arguments (line 199)
        # Getting the type of 's' (line 199)
        s_90032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 's', False)
        keyword_90033 = s_90032
        # Getting the type of 'per' (line 199)
        per_90034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'per', False)
        keyword_90035 = per_90034
        # Getting the type of 'k' (line 199)
        k_90036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 49), 'k', False)
        keyword_90037 = k_90036
        int_90038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 56), 'int')
        keyword_90039 = int_90038
        kwargs_90040 = {'nest': keyword_90039, 's': keyword_90033, 'k': keyword_90037, 'per': keyword_90035}
        # Getting the type of 'splprep' (line 199)
        splprep_90028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'splprep', False)
        # Calling splprep(args, kwargs) (line 199)
        splprep_call_result_90041 = invoke(stypy.reporting.localization.Localization(__file__, 199, 21), splprep_90028, *[list_90029], **kwargs_90040)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___90042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), splprep_call_result_90041, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_90043 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), getitem___90042, int_90027)
        
        # Assigning a type to the variable 'tuple_var_assignment_88953' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'tuple_var_assignment_88953', subscript_call_result_90043)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_88952' (line 199)
        tuple_var_assignment_88952_90044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'tuple_var_assignment_88952')
        # Assigning a type to the variable 'tckp' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'tckp', tuple_var_assignment_88952_90044)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_88953' (line 199)
        tuple_var_assignment_88953_90045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'tuple_var_assignment_88953')
        # Assigning a type to the variable 'u' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'u', tuple_var_assignment_88953_90045)
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to splrep(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'x' (line 200)
        x_90047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'x', False)
        # Getting the type of 'v' (line 200)
        v_90048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'v', False)
        # Processing the call keyword arguments (line 200)
        # Getting the type of 's' (line 200)
        s_90049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 31), 's', False)
        keyword_90050 = s_90049
        # Getting the type of 'per' (line 200)
        per_90051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'per', False)
        keyword_90052 = per_90051
        # Getting the type of 'k' (line 200)
        k_90053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 43), 'k', False)
        keyword_90054 = k_90053
        kwargs_90055 = {'s': keyword_90050, 'k': keyword_90054, 'per': keyword_90052}
        # Getting the type of 'splrep' (line 200)
        splrep_90046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'splrep', False)
        # Calling splrep(args, kwargs) (line 200)
        splrep_call_result_90056 = invoke(stypy.reporting.localization.Localization(__file__, 200, 18), splrep_90046, *[x_90047, v_90048], **kwargs_90055)
        
        # Assigning a type to the variable 'tck' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'tck', splrep_call_result_90056)
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to splev(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'dx' (line 201)
        dx_90058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'dx', False)
        # Getting the type of 'tckp' (line 201)
        tckp_90059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), 'tckp', False)
        # Processing the call keyword arguments (line 201)
        kwargs_90060 = {}
        # Getting the type of 'splev' (line 201)
        splev_90057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'splev', False)
        # Calling splev(args, kwargs) (line 201)
        splev_call_result_90061 = invoke(stypy.reporting.localization.Localization(__file__, 201, 17), splev_90057, *[dx_90058, tckp_90059], **kwargs_90060)
        
        # Assigning a type to the variable 'uv' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'uv', splev_call_result_90061)
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to abs(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining the type of the subscript
        int_90063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'int')
        # Getting the type of 'uv' (line 202)
        uv_90064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'uv', False)
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___90065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 23), uv_90064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_90066 = invoke(stypy.reporting.localization.Localization(__file__, 202, 23), getitem___90065, int_90063)
        
        
        # Call to f(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining the type of the subscript
        int_90068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'int')
        # Getting the type of 'uv' (line 202)
        uv_90069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'uv', False)
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___90070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 31), uv_90069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_90071 = invoke(stypy.reporting.localization.Localization(__file__, 202, 31), getitem___90070, int_90068)
        
        # Processing the call keyword arguments (line 202)
        kwargs_90072 = {}
        # Getting the type of 'f' (line 202)
        f_90067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'f', False)
        # Calling f(args, kwargs) (line 202)
        f_call_result_90073 = invoke(stypy.reporting.localization.Localization(__file__, 202, 29), f_90067, *[subscript_call_result_90071], **kwargs_90072)
        
        # Applying the binary operator '-' (line 202)
        result_sub_90074 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 23), '-', subscript_call_result_90066, f_call_result_90073)
        
        # Processing the call keyword arguments (line 202)
        kwargs_90075 = {}
        # Getting the type of 'abs' (line 202)
        abs_90062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'abs', False)
        # Calling abs(args, kwargs) (line 202)
        abs_call_result_90076 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), abs_90062, *[result_sub_90074], **kwargs_90075)
        
        # Assigning a type to the variable 'err1' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'err1', abs_call_result_90076)
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to abs(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to splev(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        int_90079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 32), 'int')
        # Getting the type of 'uv' (line 203)
        uv_90080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'uv', False)
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___90081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 29), uv_90080, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_90082 = invoke(stypy.reporting.localization.Localization(__file__, 203, 29), getitem___90081, int_90079)
        
        # Getting the type of 'tck' (line 203)
        tck_90083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'tck', False)
        # Processing the call keyword arguments (line 203)
        kwargs_90084 = {}
        # Getting the type of 'splev' (line 203)
        splev_90078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'splev', False)
        # Calling splev(args, kwargs) (line 203)
        splev_call_result_90085 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), splev_90078, *[subscript_call_result_90082, tck_90083], **kwargs_90084)
        
        
        # Call to f(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        int_90087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 45), 'int')
        # Getting the type of 'uv' (line 203)
        uv_90088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'uv', False)
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___90089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 42), uv_90088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_90090 = invoke(stypy.reporting.localization.Localization(__file__, 203, 42), getitem___90089, int_90087)
        
        # Processing the call keyword arguments (line 203)
        kwargs_90091 = {}
        # Getting the type of 'f' (line 203)
        f_90086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'f', False)
        # Calling f(args, kwargs) (line 203)
        f_call_result_90092 = invoke(stypy.reporting.localization.Localization(__file__, 203, 40), f_90086, *[subscript_call_result_90090], **kwargs_90091)
        
        # Applying the binary operator '-' (line 203)
        result_sub_90093 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 23), '-', splev_call_result_90085, f_call_result_90092)
        
        # Processing the call keyword arguments (line 203)
        kwargs_90094 = {}
        # Getting the type of 'abs' (line 203)
        abs_90077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 19), 'abs', False)
        # Calling abs(args, kwargs) (line 203)
        abs_call_result_90095 = invoke(stypy.reporting.localization.Localization(__file__, 203, 19), abs_90077, *[result_sub_90093], **kwargs_90094)
        
        # Assigning a type to the variable 'err2' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'err2', abs_call_result_90095)
        
        # Call to assert_(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Getting the type of 'err1' (line 204)
        err1_90097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'err1', False)
        float_90098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'float')
        # Applying the binary operator '<' (line 204)
        result_lt_90099 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 20), '<', err1_90097, float_90098)
        
        # Processing the call keyword arguments (line 204)
        kwargs_90100 = {}
        # Getting the type of 'assert_' (line 204)
        assert__90096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 204)
        assert__call_result_90101 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), assert__90096, *[result_lt_90099], **kwargs_90100)
        
        
        # Call to assert_(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Getting the type of 'err2' (line 205)
        err2_90103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'err2', False)
        float_90104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 27), 'float')
        # Applying the binary operator '<' (line 205)
        result_lt_90105 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 20), '<', err2_90103, float_90104)
        
        # Processing the call keyword arguments (line 205)
        kwargs_90106 = {}
        # Getting the type of 'assert_' (line 205)
        assert__90102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 205)
        assert__call_result_90107 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), assert__90102, *[result_lt_90105], **kwargs_90106)
        
        
        # Call to put(...): (line 206)
        # Processing the call arguments (line 206)
        str_90109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 16), 'str', '  %d  :  %s    %.1e           %.1e')
        
        # Obtaining an instance of the builtin type 'tuple' (line 207)
        tuple_90110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 207)
        # Adding element type (line 207)
        # Getting the type of 'k' (line 207)
        k_90111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), tuple_90110, k_90111)
        # Adding element type (line 207)
        
        # Call to repr(...): (line 207)
        # Processing the call arguments (line 207)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'uv' (line 207)
        uv_90118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 47), 'uv', False)
        comprehension_90119 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 27), uv_90118)
        # Assigning a type to the variable 'z' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'z', comprehension_90119)
        
        # Call to round(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'z' (line 207)
        z_90114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'z', False)
        int_90115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 35), 'int')
        # Processing the call keyword arguments (line 207)
        kwargs_90116 = {}
        # Getting the type of 'round' (line 207)
        round_90113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'round', False)
        # Calling round(args, kwargs) (line 207)
        round_call_result_90117 = invoke(stypy.reporting.localization.Localization(__file__, 207, 27), round_90113, *[z_90114, int_90115], **kwargs_90116)
        
        list_90120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 27), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 27), list_90120, round_call_result_90117)
        # Processing the call keyword arguments (line 207)
        kwargs_90121 = {}
        # Getting the type of 'repr' (line 207)
        repr_90112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'repr', False)
        # Calling repr(args, kwargs) (line 207)
        repr_call_result_90122 = invoke(stypy.reporting.localization.Localization(__file__, 207, 21), repr_90112, *[list_90120], **kwargs_90121)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), tuple_90110, repr_call_result_90122)
        # Adding element type (line 207)
        # Getting the type of 'err1' (line 208)
        err1_90123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 19), 'err1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), tuple_90110, err1_90123)
        # Adding element type (line 207)
        # Getting the type of 'err2' (line 209)
        err2_90124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'err2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 19), tuple_90110, err2_90124)
        
        # Applying the binary operator '%' (line 206)
        result_mod_90125 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 16), '%', str_90109, tuple_90110)
        
        # Processing the call keyword arguments (line 206)
        kwargs_90126 = {}
        # Getting the type of 'put' (line 206)
        put_90108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'put', False)
        # Calling put(args, kwargs) (line 206)
        put_call_result_90127 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), put_90108, *[result_mod_90125], **kwargs_90126)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to put(...): (line 210)
        # Processing the call arguments (line 210)
        str_90129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'str', 'Derivatives of parametric cubic spline at u (first function):')
        # Processing the call keyword arguments (line 210)
        kwargs_90130 = {}
        # Getting the type of 'put' (line 210)
        put_90128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'put', False)
        # Calling put(args, kwargs) (line 210)
        put_call_result_90131 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), put_90128, *[str_90129], **kwargs_90130)
        
        
        # Assigning a Num to a Name (line 211):
        
        # Assigning a Num to a Name (line 211):
        int_90132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Assigning a type to the variable 'k' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'k', int_90132)
        
        # Assigning a Call to a Tuple (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_90133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to splprep(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_90135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'x' (line 212)
        x_90136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 25), list_90135, x_90136)
        # Adding element type (line 212)
        # Getting the type of 'v' (line 212)
        v_90137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 25), list_90135, v_90137)
        
        # Processing the call keyword arguments (line 212)
        # Getting the type of 's' (line 212)
        s_90138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 's', False)
        keyword_90139 = s_90138
        # Getting the type of 'per' (line 212)
        per_90140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 39), 'per', False)
        keyword_90141 = per_90140
        # Getting the type of 'k' (line 212)
        k_90142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 45), 'k', False)
        keyword_90143 = k_90142
        int_90144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 52), 'int')
        keyword_90145 = int_90144
        kwargs_90146 = {'nest': keyword_90145, 's': keyword_90139, 'k': keyword_90143, 'per': keyword_90141}
        # Getting the type of 'splprep' (line 212)
        splprep_90134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'splprep', False)
        # Calling splprep(args, kwargs) (line 212)
        splprep_call_result_90147 = invoke(stypy.reporting.localization.Localization(__file__, 212, 17), splprep_90134, *[list_90135], **kwargs_90146)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___90148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), splprep_call_result_90147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_90149 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___90148, int_90133)
        
        # Assigning a type to the variable 'tuple_var_assignment_88954' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_88954', subscript_call_result_90149)
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        int_90150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
        
        # Call to splprep(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_90152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        # Getting the type of 'x' (line 212)
        x_90153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 25), list_90152, x_90153)
        # Adding element type (line 212)
        # Getting the type of 'v' (line 212)
        v_90154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 25), list_90152, v_90154)
        
        # Processing the call keyword arguments (line 212)
        # Getting the type of 's' (line 212)
        s_90155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 's', False)
        keyword_90156 = s_90155
        # Getting the type of 'per' (line 212)
        per_90157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 39), 'per', False)
        keyword_90158 = per_90157
        # Getting the type of 'k' (line 212)
        k_90159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 45), 'k', False)
        keyword_90160 = k_90159
        int_90161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 52), 'int')
        keyword_90162 = int_90161
        kwargs_90163 = {'nest': keyword_90162, 's': keyword_90156, 'k': keyword_90160, 'per': keyword_90158}
        # Getting the type of 'splprep' (line 212)
        splprep_90151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'splprep', False)
        # Calling splprep(args, kwargs) (line 212)
        splprep_call_result_90164 = invoke(stypy.reporting.localization.Localization(__file__, 212, 17), splprep_90151, *[list_90152], **kwargs_90163)
        
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___90165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), splprep_call_result_90164, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_90166 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___90165, int_90150)
        
        # Assigning a type to the variable 'tuple_var_assignment_88955' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_88955', subscript_call_result_90166)
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_88954' (line 212)
        tuple_var_assignment_88954_90167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_88954')
        # Assigning a type to the variable 'tckp' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tckp', tuple_var_assignment_88954_90167)
        
        # Assigning a Name to a Name (line 212):
        # Getting the type of 'tuple_var_assignment_88955' (line 212)
        tuple_var_assignment_88955_90168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_88955')
        # Assigning a type to the variable 'u' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'u', tuple_var_assignment_88955_90168)
        
        
        # Call to range(...): (line 213)
        # Processing the call arguments (line 213)
        int_90170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'int')
        # Getting the type of 'k' (line 213)
        k_90171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'k', False)
        int_90172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 27), 'int')
        # Applying the binary operator '+' (line 213)
        result_add_90173 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 25), '+', k_90171, int_90172)
        
        # Processing the call keyword arguments (line 213)
        kwargs_90174 = {}
        # Getting the type of 'range' (line 213)
        range_90169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'range', False)
        # Calling range(args, kwargs) (line 213)
        range_call_result_90175 = invoke(stypy.reporting.localization.Localization(__file__, 213, 17), range_90169, *[int_90170, result_add_90173], **kwargs_90174)
        
        # Testing the type of a for loop iterable (line 213)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 213, 8), range_call_result_90175)
        # Getting the type of the for loop variable (line 213)
        for_loop_var_90176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 213, 8), range_call_result_90175)
        # Assigning a type to the variable 'd' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'd', for_loop_var_90176)
        # SSA begins for a for statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to splev(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'dx' (line 214)
        dx_90178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'dx', False)
        # Getting the type of 'tckp' (line 214)
        tckp_90179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'tckp', False)
        # Getting the type of 'd' (line 214)
        d_90180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'd', False)
        # Processing the call keyword arguments (line 214)
        kwargs_90181 = {}
        # Getting the type of 'splev' (line 214)
        splev_90177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'splev', False)
        # Calling splev(args, kwargs) (line 214)
        splev_call_result_90182 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), splev_90177, *[dx_90178, tckp_90179, d_90180], **kwargs_90181)
        
        # Assigning a type to the variable 'uv' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'uv', splev_call_result_90182)
        
        # Call to put(...): (line 215)
        # Processing the call arguments (line 215)
        str_90184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'str', ' %s ')
        
        # Call to repr(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining the type of the subscript
        int_90186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 34), 'int')
        # Getting the type of 'uv' (line 215)
        uv_90187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'uv', False)
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___90188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 31), uv_90187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_90189 = invoke(stypy.reporting.localization.Localization(__file__, 215, 31), getitem___90188, int_90186)
        
        # Processing the call keyword arguments (line 215)
        kwargs_90190 = {}
        # Getting the type of 'repr' (line 215)
        repr_90185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 26), 'repr', False)
        # Calling repr(args, kwargs) (line 215)
        repr_call_result_90191 = invoke(stypy.reporting.localization.Localization(__file__, 215, 26), repr_90185, *[subscript_call_result_90189], **kwargs_90190)
        
        # Applying the binary operator '%' (line 215)
        result_mod_90192 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 16), '%', str_90184, repr_call_result_90191)
        
        # Processing the call keyword arguments (line 215)
        kwargs_90193 = {}
        # Getting the type of 'put' (line 215)
        put_90183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'put', False)
        # Calling put(args, kwargs) (line 215)
        put_call_result_90194 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), put_90183, *[result_mod_90192], **kwargs_90193)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_4' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_90195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_4'
        return stypy_return_type_90195


    @norecursion
    def check_5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'f2' (line 217)
        f2_90196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 23), 'f2')
        int_90197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'int')
        int_90198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'int')
        int_90199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 39), 'int')
        int_90200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'int')
        # Getting the type of 'pi' (line 217)
        pi_90201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 46), 'pi')
        # Applying the binary operator '*' (line 217)
        result_mul_90202 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 44), '*', int_90200, pi_90201)
        
        int_90203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 52), 'int')
        int_90204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'int')
        # Getting the type of 'pi' (line 217)
        pi_90205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 'pi')
        # Applying the binary operator '*' (line 217)
        result_mul_90206 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 57), '*', int_90204, pi_90205)
        
        int_90207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 65), 'int')
        int_90208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 71), 'int')
        int_90209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 76), 'int')
        defaults = [f2_90196, int_90197, int_90198, int_90199, result_mul_90202, int_90203, result_mul_90206, int_90207, int_90208, int_90209]
        # Create a new context for function 'check_5'
        module_type_store = module_type_store.open_function_context('check_5', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.check_5')
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_param_names_list', ['f', 'kx', 'ky', 'xb', 'xe', 'yb', 'ye', 'Nx', 'Ny', 's'])
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.check_5.__dict__.__setitem__('stypy_declared_arg_number', 11)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.check_5', ['f', 'kx', 'ky', 'xb', 'xe', 'yb', 'ye', 'Nx', 'Ny', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_5', localization, ['f', 'kx', 'ky', 'xb', 'xe', 'yb', 'ye', 'Nx', 'Ny', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_5(...)' code ##################

        
        # Assigning a BinOp to a Name (line 218):
        
        # Assigning a BinOp to a Name (line 218):
        # Getting the type of 'xb' (line 218)
        xb_90210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'xb')
        # Getting the type of 'xe' (line 218)
        xe_90211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'xe')
        # Getting the type of 'xb' (line 218)
        xb_90212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'xb')
        # Applying the binary operator '-' (line 218)
        result_sub_90213 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '-', xe_90211, xb_90212)
        
        
        # Call to arange(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'Nx' (line 218)
        Nx_90215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 30), 'Nx', False)
        int_90216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 33), 'int')
        # Applying the binary operator '+' (line 218)
        result_add_90217 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 30), '+', Nx_90215, int_90216)
        
        # Processing the call keyword arguments (line 218)
        # Getting the type of 'float' (line 218)
        float_90218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 41), 'float', False)
        keyword_90219 = float_90218
        kwargs_90220 = {'dtype': keyword_90219}
        # Getting the type of 'arange' (line 218)
        arange_90214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'arange', False)
        # Calling arange(args, kwargs) (line 218)
        arange_call_result_90221 = invoke(stypy.reporting.localization.Localization(__file__, 218, 23), arange_90214, *[result_add_90217], **kwargs_90220)
        
        # Applying the binary operator '*' (line 218)
        result_mul_90222 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 15), '*', result_sub_90213, arange_call_result_90221)
        
        
        # Call to float(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'Nx' (line 218)
        Nx_90224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 54), 'Nx', False)
        # Processing the call keyword arguments (line 218)
        kwargs_90225 = {}
        # Getting the type of 'float' (line 218)
        float_90223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 48), 'float', False)
        # Calling float(args, kwargs) (line 218)
        float_call_result_90226 = invoke(stypy.reporting.localization.Localization(__file__, 218, 48), float_90223, *[Nx_90224], **kwargs_90225)
        
        # Applying the binary operator 'div' (line 218)
        result_div_90227 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 47), 'div', result_mul_90222, float_call_result_90226)
        
        # Applying the binary operator '+' (line 218)
        result_add_90228 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 12), '+', xb_90210, result_div_90227)
        
        # Assigning a type to the variable 'x' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'x', result_add_90228)
        
        # Assigning a BinOp to a Name (line 219):
        
        # Assigning a BinOp to a Name (line 219):
        # Getting the type of 'yb' (line 219)
        yb_90229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'yb')
        # Getting the type of 'ye' (line 219)
        ye_90230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'ye')
        # Getting the type of 'yb' (line 219)
        yb_90231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'yb')
        # Applying the binary operator '-' (line 219)
        result_sub_90232 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '-', ye_90230, yb_90231)
        
        
        # Call to arange(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'Ny' (line 219)
        Ny_90234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'Ny', False)
        int_90235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 33), 'int')
        # Applying the binary operator '+' (line 219)
        result_add_90236 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 30), '+', Ny_90234, int_90235)
        
        # Processing the call keyword arguments (line 219)
        # Getting the type of 'float' (line 219)
        float_90237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 41), 'float', False)
        keyword_90238 = float_90237
        kwargs_90239 = {'dtype': keyword_90238}
        # Getting the type of 'arange' (line 219)
        arange_90233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 23), 'arange', False)
        # Calling arange(args, kwargs) (line 219)
        arange_call_result_90240 = invoke(stypy.reporting.localization.Localization(__file__, 219, 23), arange_90233, *[result_add_90236], **kwargs_90239)
        
        # Applying the binary operator '*' (line 219)
        result_mul_90241 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), '*', result_sub_90232, arange_call_result_90240)
        
        
        # Call to float(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'Ny' (line 219)
        Ny_90243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 54), 'Ny', False)
        # Processing the call keyword arguments (line 219)
        kwargs_90244 = {}
        # Getting the type of 'float' (line 219)
        float_90242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 48), 'float', False)
        # Calling float(args, kwargs) (line 219)
        float_call_result_90245 = invoke(stypy.reporting.localization.Localization(__file__, 219, 48), float_90242, *[Ny_90243], **kwargs_90244)
        
        # Applying the binary operator 'div' (line 219)
        result_div_90246 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 47), 'div', result_mul_90241, float_call_result_90245)
        
        # Applying the binary operator '+' (line 219)
        result_add_90247 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 12), '+', yb_90229, result_div_90246)
        
        # Assigning a type to the variable 'y' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'y', result_add_90247)
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to makepairs(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'x' (line 220)
        x_90249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'x', False)
        # Getting the type of 'y' (line 220)
        y_90250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'y', False)
        # Processing the call keyword arguments (line 220)
        kwargs_90251 = {}
        # Getting the type of 'makepairs' (line 220)
        makepairs_90248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'makepairs', False)
        # Calling makepairs(args, kwargs) (line 220)
        makepairs_call_result_90252 = invoke(stypy.reporting.localization.Localization(__file__, 220, 13), makepairs_90248, *[x_90249, y_90250], **kwargs_90251)
        
        # Assigning a type to the variable 'xy' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'xy', makepairs_call_result_90252)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to bisplrep(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        int_90254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'int')
        # Getting the type of 'xy' (line 221)
        xy_90255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___90256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 23), xy_90255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_90257 = invoke(stypy.reporting.localization.Localization(__file__, 221, 23), getitem___90256, int_90254)
        
        
        # Obtaining the type of the subscript
        int_90258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 32), 'int')
        # Getting the type of 'xy' (line 221)
        xy_90259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___90260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), xy_90259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_90261 = invoke(stypy.reporting.localization.Localization(__file__, 221, 29), getitem___90260, int_90258)
        
        
        # Call to f(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        int_90263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 40), 'int')
        # Getting the type of 'xy' (line 221)
        xy_90264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___90265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 37), xy_90264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_90266 = invoke(stypy.reporting.localization.Localization(__file__, 221, 37), getitem___90265, int_90263)
        
        
        # Obtaining the type of the subscript
        int_90267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 46), 'int')
        # Getting the type of 'xy' (line 221)
        xy_90268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 'xy', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___90269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 43), xy_90268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_90270 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), getitem___90269, int_90267)
        
        # Processing the call keyword arguments (line 221)
        kwargs_90271 = {}
        # Getting the type of 'f' (line 221)
        f_90262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 'f', False)
        # Calling f(args, kwargs) (line 221)
        f_call_result_90272 = invoke(stypy.reporting.localization.Localization(__file__, 221, 35), f_90262, *[subscript_call_result_90266, subscript_call_result_90270], **kwargs_90271)
        
        # Processing the call keyword arguments (line 221)
        # Getting the type of 's' (line 221)
        s_90273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 52), 's', False)
        keyword_90274 = s_90273
        # Getting the type of 'kx' (line 221)
        kx_90275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'kx', False)
        keyword_90276 = kx_90275
        # Getting the type of 'ky' (line 221)
        ky_90277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 63), 'ky', False)
        keyword_90278 = ky_90277
        kwargs_90279 = {'s': keyword_90274, 'kx': keyword_90276, 'ky': keyword_90278}
        # Getting the type of 'bisplrep' (line 221)
        bisplrep_90253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'bisplrep', False)
        # Calling bisplrep(args, kwargs) (line 221)
        bisplrep_call_result_90280 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), bisplrep_90253, *[subscript_call_result_90257, subscript_call_result_90261, f_call_result_90272], **kwargs_90279)
        
        # Assigning a type to the variable 'tck' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tck', bisplrep_call_result_90280)
        
        # Assigning a List to a Name (line 222):
        
        # Assigning a List to a Name (line 222):
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_90281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        
        # Obtaining the type of the subscript
        # Getting the type of 'kx' (line 222)
        kx_90282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'kx')
        
        # Getting the type of 'kx' (line 222)
        kx_90283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'kx')
        # Applying the 'usub' unary operator (line 222)
        result___neg___90284 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 24), 'usub', kx_90283)
        
        slice_90285 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 222, 14), kx_90282, result___neg___90284, None)
        
        # Obtaining the type of the subscript
        int_90286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 18), 'int')
        # Getting the type of 'tck' (line 222)
        tck_90287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'tck')
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___90288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 14), tck_90287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_90289 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), getitem___90288, int_90286)
        
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___90290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 14), subscript_call_result_90289, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_90291 = invoke(stypy.reporting.localization.Localization(__file__, 222, 14), getitem___90290, slice_90285)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 13), list_90281, subscript_call_result_90291)
        # Adding element type (line 222)
        
        # Obtaining the type of the subscript
        # Getting the type of 'ky' (line 222)
        ky_90292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'ky')
        
        # Getting the type of 'ky' (line 222)
        ky_90293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'ky')
        # Applying the 'usub' unary operator (line 222)
        result___neg___90294 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 39), 'usub', ky_90293)
        
        slice_90295 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 222, 29), ky_90292, result___neg___90294, None)
        
        # Obtaining the type of the subscript
        int_90296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 33), 'int')
        # Getting the type of 'tck' (line 222)
        tck_90297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 29), 'tck')
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___90298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 29), tck_90297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_90299 = invoke(stypy.reporting.localization.Localization(__file__, 222, 29), getitem___90298, int_90296)
        
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___90300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 29), subscript_call_result_90299, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_90301 = invoke(stypy.reporting.localization.Localization(__file__, 222, 29), getitem___90300, slice_90295)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 13), list_90281, subscript_call_result_90301)
        
        # Assigning a type to the variable 'tt' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'tt', list_90281)
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to makepairs(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining the type of the subscript
        int_90303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 26), 'int')
        # Getting the type of 'tt' (line 223)
        tt_90304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___90305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 23), tt_90304, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_90306 = invoke(stypy.reporting.localization.Localization(__file__, 223, 23), getitem___90305, int_90303)
        
        
        # Obtaining the type of the subscript
        int_90307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 32), 'int')
        # Getting the type of 'tt' (line 223)
        tt_90308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 29), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___90309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 29), tt_90308, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_90310 = invoke(stypy.reporting.localization.Localization(__file__, 223, 29), getitem___90309, int_90307)
        
        # Processing the call keyword arguments (line 223)
        kwargs_90311 = {}
        # Getting the type of 'makepairs' (line 223)
        makepairs_90302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 13), 'makepairs', False)
        # Calling makepairs(args, kwargs) (line 223)
        makepairs_call_result_90312 = invoke(stypy.reporting.localization.Localization(__file__, 223, 13), makepairs_90302, *[subscript_call_result_90306, subscript_call_result_90310], **kwargs_90311)
        
        # Assigning a type to the variable 't2' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 't2', makepairs_call_result_90312)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to bisplev(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Obtaining the type of the subscript
        int_90314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 24), 'int')
        # Getting the type of 'tt' (line 224)
        tt_90315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___90316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 21), tt_90315, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_90317 = invoke(stypy.reporting.localization.Localization(__file__, 224, 21), getitem___90316, int_90314)
        
        
        # Obtaining the type of the subscript
        int_90318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 30), 'int')
        # Getting the type of 'tt' (line 224)
        tt_90319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___90320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), tt_90319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_90321 = invoke(stypy.reporting.localization.Localization(__file__, 224, 27), getitem___90320, int_90318)
        
        # Getting the type of 'tck' (line 224)
        tck_90322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'tck', False)
        # Processing the call keyword arguments (line 224)
        kwargs_90323 = {}
        # Getting the type of 'bisplev' (line 224)
        bisplev_90313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'bisplev', False)
        # Calling bisplev(args, kwargs) (line 224)
        bisplev_call_result_90324 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), bisplev_90313, *[subscript_call_result_90317, subscript_call_result_90321, tck_90322], **kwargs_90323)
        
        # Assigning a type to the variable 'v1' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'v1', bisplev_call_result_90324)
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to f2(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Obtaining the type of the subscript
        int_90326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'int')
        # Getting the type of 't2' (line 225)
        t2_90327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 't2', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___90328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), t2_90327, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_90329 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), getitem___90328, int_90326)
        
        
        # Obtaining the type of the subscript
        int_90330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'int')
        # Getting the type of 't2' (line 225)
        t2_90331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 't2', False)
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___90332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), t2_90331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_90333 = invoke(stypy.reporting.localization.Localization(__file__, 225, 22), getitem___90332, int_90330)
        
        # Processing the call keyword arguments (line 225)
        kwargs_90334 = {}
        # Getting the type of 'f2' (line 225)
        f2_90325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 13), 'f2', False)
        # Calling f2(args, kwargs) (line 225)
        f2_call_result_90335 = invoke(stypy.reporting.localization.Localization(__file__, 225, 13), f2_90325, *[subscript_call_result_90329, subscript_call_result_90333], **kwargs_90334)
        
        # Assigning a type to the variable 'v2' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'v2', f2_call_result_90335)
        
        # Assigning a Tuple to a Attribute (line 226):
        
        # Assigning a Tuple to a Attribute (line 226):
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_90336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        
        # Call to len(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining the type of the subscript
        int_90338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 26), 'int')
        # Getting the type of 'tt' (line 226)
        tt_90339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___90340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 23), tt_90339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_90341 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), getitem___90340, int_90338)
        
        # Processing the call keyword arguments (line 226)
        kwargs_90342 = {}
        # Getting the type of 'len' (line 226)
        len_90337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'len', False)
        # Calling len(args, kwargs) (line 226)
        len_call_result_90343 = invoke(stypy.reporting.localization.Localization(__file__, 226, 19), len_90337, *[subscript_call_result_90341], **kwargs_90342)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_90336, len_call_result_90343)
        # Adding element type (line 226)
        
        # Call to len(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining the type of the subscript
        int_90345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 37), 'int')
        # Getting the type of 'tt' (line 226)
        tt_90346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'tt', False)
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___90347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 34), tt_90346, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_90348 = invoke(stypy.reporting.localization.Localization(__file__, 226, 34), getitem___90347, int_90345)
        
        # Processing the call keyword arguments (line 226)
        kwargs_90349 = {}
        # Getting the type of 'len' (line 226)
        len_90344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'len', False)
        # Calling len(args, kwargs) (line 226)
        len_call_result_90350 = invoke(stypy.reporting.localization.Localization(__file__, 226, 30), len_90344, *[subscript_call_result_90348], **kwargs_90349)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_90336, len_call_result_90350)
        
        # Getting the type of 'v2' (line 226)
        v2_90351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'v2')
        # Setting the type of the member 'shape' of a type (line 226)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), v2_90351, 'shape', tuple_90336)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to norm2(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to ravel(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'v1' (line 227)
        v1_90354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'v1', False)
        # Getting the type of 'v2' (line 227)
        v2_90355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'v2', False)
        # Applying the binary operator '-' (line 227)
        result_sub_90356 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 26), '-', v1_90354, v2_90355)
        
        # Processing the call keyword arguments (line 227)
        kwargs_90357 = {}
        # Getting the type of 'ravel' (line 227)
        ravel_90353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'ravel', False)
        # Calling ravel(args, kwargs) (line 227)
        ravel_call_result_90358 = invoke(stypy.reporting.localization.Localization(__file__, 227, 20), ravel_90353, *[result_sub_90356], **kwargs_90357)
        
        # Processing the call keyword arguments (line 227)
        kwargs_90359 = {}
        # Getting the type of 'norm2' (line 227)
        norm2_90352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 14), 'norm2', False)
        # Calling norm2(args, kwargs) (line 227)
        norm2_call_result_90360 = invoke(stypy.reporting.localization.Localization(__file__, 227, 14), norm2_90352, *[ravel_call_result_90358], **kwargs_90359)
        
        # Assigning a type to the variable 'err' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'err', norm2_call_result_90360)
        
        # Call to assert_(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Getting the type of 'err' (line 228)
        err_90362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'err', False)
        float_90363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'float')
        # Applying the binary operator '<' (line 228)
        result_lt_90364 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 16), '<', err_90362, float_90363)
        
        # Getting the type of 'err' (line 228)
        err_90365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'err', False)
        # Processing the call keyword arguments (line 228)
        kwargs_90366 = {}
        # Getting the type of 'assert_' (line 228)
        assert__90361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 228)
        assert__call_result_90367 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert__90361, *[result_lt_90364, err_90365], **kwargs_90366)
        
        
        # Call to put(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'err' (line 229)
        err_90369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'err', False)
        # Processing the call keyword arguments (line 229)
        kwargs_90370 = {}
        # Getting the type of 'put' (line 229)
        put_90368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'put', False)
        # Calling put(args, kwargs) (line 229)
        put_call_result_90371 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), put_90368, *[err_90369], **kwargs_90370)
        
        
        # ################# End of 'check_5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_5' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_90372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_5'
        return stypy_return_type_90372


    @norecursion
    def test_smoke_splrep_splev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoke_splrep_splev'
        module_type_store = module_type_store.open_function_context('test_smoke_splrep_splev', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.test_smoke_splrep_splev')
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.test_smoke_splrep_splev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.test_smoke_splrep_splev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoke_splrep_splev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoke_splrep_splev(...)' code ##################

        
        # Call to put(...): (line 232)
        # Processing the call arguments (line 232)
        str_90374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'str', '***************** splrep/splev')
        # Processing the call keyword arguments (line 232)
        kwargs_90375 = {}
        # Getting the type of 'put' (line 232)
        put_90373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'put', False)
        # Calling put(args, kwargs) (line 232)
        put_call_result_90376 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), put_90373, *[str_90374], **kwargs_90375)
        
        
        # Call to check_1(...): (line 233)
        # Processing the call keyword arguments (line 233)
        float_90379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 23), 'float')
        keyword_90380 = float_90379
        kwargs_90381 = {'s': keyword_90380}
        # Getting the type of 'self' (line 233)
        self_90377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 233)
        check_1_90378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_90377, 'check_1')
        # Calling check_1(args, kwargs) (line 233)
        check_1_call_result_90382 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), check_1_90378, *[], **kwargs_90381)
        
        
        # Call to check_1(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_90385 = {}
        # Getting the type of 'self' (line 234)
        self_90383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 234)
        check_1_90384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_90383, 'check_1')
        # Calling check_1(args, kwargs) (line 234)
        check_1_call_result_90386 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), check_1_90384, *[], **kwargs_90385)
        
        
        # Call to check_1(...): (line 235)
        # Processing the call keyword arguments (line 235)
        int_90389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'int')
        keyword_90390 = int_90389
        kwargs_90391 = {'at': keyword_90390}
        # Getting the type of 'self' (line 235)
        self_90387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 235)
        check_1_90388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_90387, 'check_1')
        # Calling check_1(args, kwargs) (line 235)
        check_1_call_result_90392 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), check_1_90388, *[], **kwargs_90391)
        
        
        # Call to check_1(...): (line 236)
        # Processing the call keyword arguments (line 236)
        int_90395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 25), 'int')
        keyword_90396 = int_90395
        kwargs_90397 = {'per': keyword_90396}
        # Getting the type of 'self' (line 236)
        self_90393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 236)
        check_1_90394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_90393, 'check_1')
        # Calling check_1(args, kwargs) (line 236)
        check_1_call_result_90398 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), check_1_90394, *[], **kwargs_90397)
        
        
        # Call to check_1(...): (line 237)
        # Processing the call keyword arguments (line 237)
        int_90401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 25), 'int')
        keyword_90402 = int_90401
        int_90403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 30), 'int')
        keyword_90404 = int_90403
        kwargs_90405 = {'at': keyword_90404, 'per': keyword_90402}
        # Getting the type of 'self' (line 237)
        self_90399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 237)
        check_1_90400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_90399, 'check_1')
        # Calling check_1(args, kwargs) (line 237)
        check_1_call_result_90406 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), check_1_90400, *[], **kwargs_90405)
        
        
        # Call to check_1(...): (line 238)
        # Processing the call keyword arguments (line 238)
        float_90409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'float')
        # Getting the type of 'pi' (line 238)
        pi_90410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'pi', False)
        # Applying the binary operator '*' (line 238)
        result_mul_90411 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 23), '*', float_90409, pi_90410)
        
        keyword_90412 = result_mul_90411
        kwargs_90413 = {'b': keyword_90412}
        # Getting the type of 'self' (line 238)
        self_90407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 238)
        check_1_90408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_90407, 'check_1')
        # Calling check_1(args, kwargs) (line 238)
        check_1_call_result_90414 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), check_1_90408, *[], **kwargs_90413)
        
        
        # Call to check_1(...): (line 239)
        # Processing the call keyword arguments (line 239)
        float_90417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 23), 'float')
        # Getting the type of 'pi' (line 239)
        pi_90418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'pi', False)
        # Applying the binary operator '*' (line 239)
        result_mul_90419 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 23), '*', float_90417, pi_90418)
        
        keyword_90420 = result_mul_90419
        int_90421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'int')
        # Getting the type of 'pi' (line 239)
        pi_90422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'pi', False)
        # Applying the binary operator '*' (line 239)
        result_mul_90423 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 33), '*', int_90421, pi_90422)
        
        keyword_90424 = result_mul_90423
        int_90425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 42), 'int')
        keyword_90426 = int_90425
        float_90427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 46), 'float')
        keyword_90428 = float_90427
        kwargs_90429 = {'s': keyword_90428, 'b': keyword_90420, 'per': keyword_90426, 'xe': keyword_90424}
        # Getting the type of 'self' (line 239)
        self_90415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'self', False)
        # Obtaining the member 'check_1' of a type (line 239)
        check_1_90416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), self_90415, 'check_1')
        # Calling check_1(args, kwargs) (line 239)
        check_1_call_result_90430 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), check_1_90416, *[], **kwargs_90429)
        
        
        # ################# End of 'test_smoke_splrep_splev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoke_splrep_splev' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_90431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoke_splrep_splev'
        return stypy_return_type_90431


    @norecursion
    def test_smoke_splint_spalde(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoke_splint_spalde'
        module_type_store = module_type_store.open_function_context('test_smoke_splint_spalde', 241, 4, False)
        # Assigning a type to the variable 'self' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.test_smoke_splint_spalde')
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.test_smoke_splint_spalde.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.test_smoke_splint_spalde', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoke_splint_spalde', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoke_splint_spalde(...)' code ##################

        
        # Call to put(...): (line 242)
        # Processing the call arguments (line 242)
        str_90433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'str', '***************** splint/spalde')
        # Processing the call keyword arguments (line 242)
        kwargs_90434 = {}
        # Getting the type of 'put' (line 242)
        put_90432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'put', False)
        # Calling put(args, kwargs) (line 242)
        put_call_result_90435 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), put_90432, *[str_90433], **kwargs_90434)
        
        
        # Call to check_2(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_90438 = {}
        # Getting the type of 'self' (line 243)
        self_90436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'self', False)
        # Obtaining the member 'check_2' of a type (line 243)
        check_2_90437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), self_90436, 'check_2')
        # Calling check_2(args, kwargs) (line 243)
        check_2_call_result_90439 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), check_2_90437, *[], **kwargs_90438)
        
        
        # Call to check_2(...): (line 244)
        # Processing the call keyword arguments (line 244)
        int_90442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 25), 'int')
        keyword_90443 = int_90442
        kwargs_90444 = {'per': keyword_90443}
        # Getting the type of 'self' (line 244)
        self_90440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'check_2' of a type (line 244)
        check_2_90441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_90440, 'check_2')
        # Calling check_2(args, kwargs) (line 244)
        check_2_call_result_90445 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), check_2_90441, *[], **kwargs_90444)
        
        
        # Call to check_2(...): (line 245)
        # Processing the call keyword arguments (line 245)
        float_90448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 24), 'float')
        # Getting the type of 'pi' (line 245)
        pi_90449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'pi', False)
        # Applying the binary operator '*' (line 245)
        result_mul_90450 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 24), '*', float_90448, pi_90449)
        
        keyword_90451 = result_mul_90450
        # Getting the type of 'pi' (line 245)
        pi_90452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 34), 'pi', False)
        keyword_90453 = pi_90452
        kwargs_90454 = {'ia': keyword_90451, 'ib': keyword_90453}
        # Getting the type of 'self' (line 245)
        self_90446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member 'check_2' of a type (line 245)
        check_2_90447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_90446, 'check_2')
        # Calling check_2(args, kwargs) (line 245)
        check_2_call_result_90455 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), check_2_90447, *[], **kwargs_90454)
        
        
        # Call to check_2(...): (line 246)
        # Processing the call keyword arguments (line 246)
        float_90458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 24), 'float')
        # Getting the type of 'pi' (line 246)
        pi_90459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'pi', False)
        # Applying the binary operator '*' (line 246)
        result_mul_90460 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 24), '*', float_90458, pi_90459)
        
        keyword_90461 = result_mul_90460
        # Getting the type of 'pi' (line 246)
        pi_90462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 34), 'pi', False)
        keyword_90463 = pi_90462
        int_90464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 39), 'int')
        keyword_90465 = int_90464
        kwargs_90466 = {'ia': keyword_90461, 'ib': keyword_90463, 'N': keyword_90465}
        # Getting the type of 'self' (line 246)
        self_90456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'check_2' of a type (line 246)
        check_2_90457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_90456, 'check_2')
        # Calling check_2(args, kwargs) (line 246)
        check_2_call_result_90467 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), check_2_90457, *[], **kwargs_90466)
        
        
        # ################# End of 'test_smoke_splint_spalde(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoke_splint_spalde' in the type store
        # Getting the type of 'stypy_return_type' (line 241)
        stypy_return_type_90468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90468)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoke_splint_spalde'
        return stypy_return_type_90468


    @norecursion
    def test_smoke_sproot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoke_sproot'
        module_type_store = module_type_store.open_function_context('test_smoke_sproot', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.test_smoke_sproot')
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.test_smoke_sproot.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.test_smoke_sproot', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoke_sproot', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoke_sproot(...)' code ##################

        
        # Call to put(...): (line 249)
        # Processing the call arguments (line 249)
        str_90470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 12), 'str', '***************** sproot')
        # Processing the call keyword arguments (line 249)
        kwargs_90471 = {}
        # Getting the type of 'put' (line 249)
        put_90469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'put', False)
        # Calling put(args, kwargs) (line 249)
        put_call_result_90472 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), put_90469, *[str_90470], **kwargs_90471)
        
        
        # Call to check_3(...): (line 250)
        # Processing the call keyword arguments (line 250)
        float_90475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 23), 'float')
        keyword_90476 = float_90475
        int_90477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 29), 'int')
        keyword_90478 = int_90477
        kwargs_90479 = {'a': keyword_90476, 'b': keyword_90478}
        # Getting the type of 'self' (line 250)
        self_90473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'check_3' of a type (line 250)
        check_3_90474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_90473, 'check_3')
        # Calling check_3(args, kwargs) (line 250)
        check_3_call_result_90480 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), check_3_90474, *[], **kwargs_90479)
        
        
        # ################# End of 'test_smoke_sproot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoke_sproot' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_90481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoke_sproot'
        return stypy_return_type_90481


    @norecursion
    def test_smoke_splprep_splrep_splev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoke_splprep_splrep_splev'
        module_type_store = module_type_store.open_function_context('test_smoke_splprep_splrep_splev', 252, 4, False)
        # Assigning a type to the variable 'self' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.test_smoke_splprep_splrep_splev')
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.test_smoke_splprep_splrep_splev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.test_smoke_splprep_splrep_splev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoke_splprep_splrep_splev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoke_splprep_splrep_splev(...)' code ##################

        
        # Call to put(...): (line 253)
        # Processing the call arguments (line 253)
        str_90483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 12), 'str', '***************** splprep/splrep/splev')
        # Processing the call keyword arguments (line 253)
        kwargs_90484 = {}
        # Getting the type of 'put' (line 253)
        put_90482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'put', False)
        # Calling put(args, kwargs) (line 253)
        put_call_result_90485 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), put_90482, *[str_90483], **kwargs_90484)
        
        
        # Call to check_4(...): (line 254)
        # Processing the call keyword arguments (line 254)
        kwargs_90488 = {}
        # Getting the type of 'self' (line 254)
        self_90486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'self', False)
        # Obtaining the member 'check_4' of a type (line 254)
        check_4_90487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), self_90486, 'check_4')
        # Calling check_4(args, kwargs) (line 254)
        check_4_call_result_90489 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), check_4_90487, *[], **kwargs_90488)
        
        
        # Call to check_4(...): (line 255)
        # Processing the call keyword arguments (line 255)
        int_90492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'int')
        keyword_90493 = int_90492
        kwargs_90494 = {'N': keyword_90493}
        # Getting the type of 'self' (line 255)
        self_90490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'self', False)
        # Obtaining the member 'check_4' of a type (line 255)
        check_4_90491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), self_90490, 'check_4')
        # Calling check_4(args, kwargs) (line 255)
        check_4_call_result_90495 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), check_4_90491, *[], **kwargs_90494)
        
        
        # ################# End of 'test_smoke_splprep_splrep_splev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoke_splprep_splrep_splev' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_90496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoke_splprep_splrep_splev'
        return stypy_return_type_90496


    @norecursion
    def test_smoke_bisplrep_bisplev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoke_bisplrep_bisplev'
        module_type_store = module_type_store.open_function_context('test_smoke_bisplrep_bisplev', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_localization', localization)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_function_name', 'TestSmokeTests.test_smoke_bisplrep_bisplev')
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_param_names_list', [])
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSmokeTests.test_smoke_bisplrep_bisplev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.test_smoke_bisplrep_bisplev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoke_bisplrep_bisplev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoke_bisplrep_bisplev(...)' code ##################

        
        # Call to put(...): (line 258)
        # Processing the call arguments (line 258)
        str_90498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 12), 'str', '***************** bisplev')
        # Processing the call keyword arguments (line 258)
        kwargs_90499 = {}
        # Getting the type of 'put' (line 258)
        put_90497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'put', False)
        # Calling put(args, kwargs) (line 258)
        put_call_result_90500 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), put_90497, *[str_90498], **kwargs_90499)
        
        
        # Call to check_5(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_90503 = {}
        # Getting the type of 'self' (line 259)
        self_90501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'check_5' of a type (line 259)
        check_5_90502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_90501, 'check_5')
        # Calling check_5(args, kwargs) (line 259)
        check_5_call_result_90504 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), check_5_90502, *[], **kwargs_90503)
        
        
        # ################# End of 'test_smoke_bisplrep_bisplev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoke_bisplrep_bisplev' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_90505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoke_bisplrep_bisplev'
        return stypy_return_type_90505


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 69, 0, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSmokeTests.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSmokeTests' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'TestSmokeTests', TestSmokeTests)
# Declaration of the 'TestSplev' class

class TestSplev(object, ):

    @norecursion
    def test_1d_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d_shape'
        module_type_store = module_type_store.open_function_context('test_1d_shape', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_localization', localization)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_function_name', 'TestSplev.test_1d_shape')
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplev.test_1d_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplev.test_1d_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d_shape(...)' code ##################

        
        # Assigning a List to a Name (line 264):
        
        # Assigning a List to a Name (line 264):
        
        # Obtaining an instance of the builtin type 'list' (line 264)
        list_90506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 264)
        # Adding element type (line 264)
        int_90507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 12), list_90506, int_90507)
        # Adding element type (line 264)
        int_90508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 12), list_90506, int_90508)
        # Adding element type (line 264)
        int_90509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 12), list_90506, int_90509)
        # Adding element type (line 264)
        int_90510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 12), list_90506, int_90510)
        # Adding element type (line 264)
        int_90511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 12), list_90506, int_90511)
        
        # Assigning a type to the variable 'x' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'x', list_90506)
        
        # Assigning a List to a Name (line 265):
        
        # Assigning a List to a Name (line 265):
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_90512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        # Adding element type (line 265)
        int_90513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), list_90512, int_90513)
        # Adding element type (line 265)
        int_90514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), list_90512, int_90514)
        # Adding element type (line 265)
        int_90515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), list_90512, int_90515)
        # Adding element type (line 265)
        int_90516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), list_90512, int_90516)
        # Adding element type (line 265)
        int_90517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 12), list_90512, int_90517)
        
        # Assigning a type to the variable 'y' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'y', list_90512)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to splrep(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_90519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'x', False)
        # Getting the type of 'y' (line 266)
        y_90520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'y', False)
        # Processing the call keyword arguments (line 266)
        kwargs_90521 = {}
        # Getting the type of 'splrep' (line 266)
        splrep_90518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'splrep', False)
        # Calling splrep(args, kwargs) (line 266)
        splrep_call_result_90522 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), splrep_90518, *[x_90519, y_90520], **kwargs_90521)
        
        # Assigning a type to the variable 'tck' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tck', splrep_call_result_90522)
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to splev(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_90524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_90525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 18), list_90524, int_90525)
        
        # Getting the type of 'tck' (line 267)
        tck_90526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'tck', False)
        # Processing the call keyword arguments (line 267)
        kwargs_90527 = {}
        # Getting the type of 'splev' (line 267)
        splev_90523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'splev', False)
        # Calling splev(args, kwargs) (line 267)
        splev_call_result_90528 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), splev_90523, *[list_90524, tck_90526], **kwargs_90527)
        
        # Assigning a type to the variable 'z' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'z', splev_call_result_90528)
        
        # Call to assert_equal(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'z' (line 268)
        z_90530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 21), 'z', False)
        # Obtaining the member 'shape' of a type (line 268)
        shape_90531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 21), z_90530, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 268)
        tuple_90532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 268)
        # Adding element type (line 268)
        int_90533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 31), tuple_90532, int_90533)
        
        # Processing the call keyword arguments (line 268)
        kwargs_90534 = {}
        # Getting the type of 'assert_equal' (line 268)
        assert_equal_90529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 268)
        assert_equal_call_result_90535 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assert_equal_90529, *[shape_90531, tuple_90532], **kwargs_90534)
        
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to splev(...): (line 269)
        # Processing the call arguments (line 269)
        int_90537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 18), 'int')
        # Getting the type of 'tck' (line 269)
        tck_90538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'tck', False)
        # Processing the call keyword arguments (line 269)
        kwargs_90539 = {}
        # Getting the type of 'splev' (line 269)
        splev_90536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'splev', False)
        # Calling splev(args, kwargs) (line 269)
        splev_call_result_90540 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), splev_90536, *[int_90537, tck_90538], **kwargs_90539)
        
        # Assigning a type to the variable 'z' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'z', splev_call_result_90540)
        
        # Call to assert_equal(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'z' (line 270)
        z_90542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 21), 'z', False)
        # Obtaining the member 'shape' of a type (line 270)
        shape_90543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 21), z_90542, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_90544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        
        # Processing the call keyword arguments (line 270)
        kwargs_90545 = {}
        # Getting the type of 'assert_equal' (line 270)
        assert_equal_90541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 270)
        assert_equal_call_result_90546 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), assert_equal_90541, *[shape_90543, tuple_90544], **kwargs_90545)
        
        
        # ################# End of 'test_1d_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_90547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d_shape'
        return stypy_return_type_90547


    @norecursion
    def test_2d_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d_shape'
        module_type_store = module_type_store.open_function_context('test_2d_shape', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_localization', localization)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_function_name', 'TestSplev.test_2d_shape')
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplev.test_2d_shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplev.test_2d_shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d_shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d_shape(...)' code ##################

        
        # Assigning a List to a Name (line 273):
        
        # Assigning a List to a Name (line 273):
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_90548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        int_90549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), list_90548, int_90549)
        # Adding element type (line 273)
        int_90550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), list_90548, int_90550)
        # Adding element type (line 273)
        int_90551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), list_90548, int_90551)
        # Adding element type (line 273)
        int_90552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), list_90548, int_90552)
        # Adding element type (line 273)
        int_90553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 12), list_90548, int_90553)
        
        # Assigning a type to the variable 'x' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'x', list_90548)
        
        # Assigning a List to a Name (line 274):
        
        # Assigning a List to a Name (line 274):
        
        # Obtaining an instance of the builtin type 'list' (line 274)
        list_90554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 274)
        # Adding element type (line 274)
        int_90555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), list_90554, int_90555)
        # Adding element type (line 274)
        int_90556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), list_90554, int_90556)
        # Adding element type (line 274)
        int_90557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), list_90554, int_90557)
        # Adding element type (line 274)
        int_90558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), list_90554, int_90558)
        # Adding element type (line 274)
        int_90559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), list_90554, int_90559)
        
        # Assigning a type to the variable 'y' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'y', list_90554)
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to splrep(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'x' (line 275)
        x_90561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'x', False)
        # Getting the type of 'y' (line 275)
        y_90562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'y', False)
        # Processing the call keyword arguments (line 275)
        kwargs_90563 = {}
        # Getting the type of 'splrep' (line 275)
        splrep_90560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 14), 'splrep', False)
        # Calling splrep(args, kwargs) (line 275)
        splrep_call_result_90564 = invoke(stypy.reporting.localization.Localization(__file__, 275, 14), splrep_90560, *[x_90561, y_90562], **kwargs_90563)
        
        # Assigning a type to the variable 'tck' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tck', splrep_call_result_90564)
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to array(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_90567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_90568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        float_90569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 22), list_90568, float_90569)
        # Adding element type (line 276)
        float_90570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 22), list_90568, float_90570)
        # Adding element type (line 276)
        float_90571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 22), list_90568, float_90571)
        # Adding element type (line 276)
        float_90572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 22), list_90568, float_90572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), list_90567, list_90568)
        # Adding element type (line 276)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_90573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        float_90574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_90573, float_90574)
        # Adding element type (line 277)
        float_90575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_90573, float_90575)
        # Adding element type (line 277)
        float_90576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_90573, float_90576)
        # Adding element type (line 277)
        float_90577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 22), list_90573, float_90577)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 21), list_90567, list_90573)
        
        # Processing the call keyword arguments (line 276)
        kwargs_90578 = {}
        # Getting the type of 'np' (line 276)
        np_90565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 276)
        array_90566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), np_90565, 'array')
        # Calling array(args, kwargs) (line 276)
        array_call_result_90579 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), array_90566, *[list_90567], **kwargs_90578)
        
        # Assigning a type to the variable 't' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 't', array_call_result_90579)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to splev(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 't' (line 278)
        t_90581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 't', False)
        # Getting the type of 'tck' (line 278)
        tck_90582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'tck', False)
        # Processing the call keyword arguments (line 278)
        kwargs_90583 = {}
        # Getting the type of 'splev' (line 278)
        splev_90580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'splev', False)
        # Calling splev(args, kwargs) (line 278)
        splev_call_result_90584 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), splev_90580, *[t_90581, tck_90582], **kwargs_90583)
        
        # Assigning a type to the variable 'z' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'z', splev_call_result_90584)
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to splev(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining the type of the subscript
        int_90586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 21), 'int')
        # Getting the type of 't' (line 279)
        t_90587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 't', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___90588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 19), t_90587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_90589 = invoke(stypy.reporting.localization.Localization(__file__, 279, 19), getitem___90588, int_90586)
        
        # Getting the type of 'tck' (line 279)
        tck_90590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 25), 'tck', False)
        # Processing the call keyword arguments (line 279)
        kwargs_90591 = {}
        # Getting the type of 'splev' (line 279)
        splev_90585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 13), 'splev', False)
        # Calling splev(args, kwargs) (line 279)
        splev_call_result_90592 = invoke(stypy.reporting.localization.Localization(__file__, 279, 13), splev_90585, *[subscript_call_result_90589, tck_90590], **kwargs_90591)
        
        # Assigning a type to the variable 'z0' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'z0', splev_call_result_90592)
        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to splev(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Obtaining the type of the subscript
        int_90594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 21), 'int')
        # Getting the type of 't' (line 280)
        t_90595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 't', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___90596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), t_90595, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_90597 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), getitem___90596, int_90594)
        
        # Getting the type of 'tck' (line 280)
        tck_90598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 25), 'tck', False)
        # Processing the call keyword arguments (line 280)
        kwargs_90599 = {}
        # Getting the type of 'splev' (line 280)
        splev_90593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'splev', False)
        # Calling splev(args, kwargs) (line 280)
        splev_call_result_90600 = invoke(stypy.reporting.localization.Localization(__file__, 280, 13), splev_90593, *[subscript_call_result_90597, tck_90598], **kwargs_90599)
        
        # Assigning a type to the variable 'z1' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'z1', splev_call_result_90600)
        
        # Call to assert_equal(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'z' (line 281)
        z_90602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'z', False)
        
        # Call to row_stack(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Obtaining an instance of the builtin type 'tuple' (line 281)
        tuple_90605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'z0' (line 281)
        z0_90606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'z0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 38), tuple_90605, z0_90606)
        # Adding element type (line 281)
        # Getting the type of 'z1' (line 281)
        z1_90607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 42), 'z1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 38), tuple_90605, z1_90607)
        
        # Processing the call keyword arguments (line 281)
        kwargs_90608 = {}
        # Getting the type of 'np' (line 281)
        np_90603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'np', False)
        # Obtaining the member 'row_stack' of a type (line 281)
        row_stack_90604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), np_90603, 'row_stack')
        # Calling row_stack(args, kwargs) (line 281)
        row_stack_call_result_90609 = invoke(stypy.reporting.localization.Localization(__file__, 281, 24), row_stack_90604, *[tuple_90605], **kwargs_90608)
        
        # Processing the call keyword arguments (line 281)
        kwargs_90610 = {}
        # Getting the type of 'assert_equal' (line 281)
        assert_equal_90601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 281)
        assert_equal_call_result_90611 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), assert_equal_90601, *[z_90602, row_stack_call_result_90609], **kwargs_90610)
        
        
        # ################# End of 'test_2d_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_90612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d_shape'
        return stypy_return_type_90612


    @norecursion
    def test_extrapolation_modes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_extrapolation_modes'
        module_type_store = module_type_store.open_function_context('test_extrapolation_modes', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_localization', localization)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_function_name', 'TestSplev.test_extrapolation_modes')
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplev.test_extrapolation_modes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplev.test_extrapolation_modes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_extrapolation_modes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_extrapolation_modes(...)' code ##################

        
        # Assigning a List to a Name (line 289):
        
        # Assigning a List to a Name (line 289):
        
        # Obtaining an instance of the builtin type 'list' (line 289)
        list_90613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 289)
        # Adding element type (line 289)
        int_90614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 12), list_90613, int_90614)
        # Adding element type (line 289)
        int_90615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 12), list_90613, int_90615)
        # Adding element type (line 289)
        int_90616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 12), list_90613, int_90616)
        
        # Assigning a type to the variable 'x' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'x', list_90613)
        
        # Assigning a List to a Name (line 290):
        
        # Assigning a List to a Name (line 290):
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_90617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        # Adding element type (line 290)
        int_90618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), list_90617, int_90618)
        # Adding element type (line 290)
        int_90619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), list_90617, int_90619)
        # Adding element type (line 290)
        int_90620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), list_90617, int_90620)
        
        # Assigning a type to the variable 'y' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'y', list_90617)
        
        # Assigning a Call to a Name (line 291):
        
        # Assigning a Call to a Name (line 291):
        
        # Call to splrep(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'x' (line 291)
        x_90622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'x', False)
        # Getting the type of 'y' (line 291)
        y_90623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'y', False)
        # Processing the call keyword arguments (line 291)
        int_90624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 29), 'int')
        keyword_90625 = int_90624
        kwargs_90626 = {'k': keyword_90625}
        # Getting the type of 'splrep' (line 291)
        splrep_90621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'splrep', False)
        # Calling splrep(args, kwargs) (line 291)
        splrep_call_result_90627 = invoke(stypy.reporting.localization.Localization(__file__, 291, 14), splrep_90621, *[x_90622, y_90623], **kwargs_90626)
        
        # Assigning a type to the variable 'tck' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tck', splrep_call_result_90627)
        
        # Assigning a List to a Name (line 293):
        
        # Assigning a List to a Name (line 293):
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_90628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_90629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_90630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 16), list_90629, int_90630)
        # Adding element type (line 293)
        int_90631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 16), list_90629, int_90631)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_90628, list_90629)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_90632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_90633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 25), list_90632, int_90633)
        # Adding element type (line 293)
        int_90634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 25), list_90632, int_90634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_90628, list_90632)
        # Adding element type (line 293)
        # Getting the type of 'None' (line 293)
        None_90635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 33), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_90628, None_90635)
        # Adding element type (line 293)
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_90636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        # Adding element type (line 293)
        int_90637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 39), list_90636, int_90637)
        # Adding element type (line 293)
        int_90638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 39), list_90636, int_90638)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 15), list_90628, list_90636)
        
        # Assigning a type to the variable 'rstl' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'rstl', list_90628)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 294)
        tuple_90639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 294)
        # Adding element type (line 294)
        int_90640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 20), tuple_90639, int_90640)
        # Adding element type (line 294)
        int_90641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 20), tuple_90639, int_90641)
        # Adding element type (line 294)
        int_90642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 20), tuple_90639, int_90642)
        
        # Testing the type of a for loop iterable (line 294)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 294, 8), tuple_90639)
        # Getting the type of the for loop variable (line 294)
        for_loop_var_90643 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 294, 8), tuple_90639)
        # Assigning a type to the variable 'ext' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'ext', for_loop_var_90643)
        # SSA begins for a for statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Call to splev(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_90646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        int_90647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 44), list_90646, int_90647)
        # Adding element type (line 295)
        int_90648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 44), list_90646, int_90648)
        
        # Getting the type of 'tck' (line 295)
        tck_90649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'tck', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'ext' (line 295)
        ext_90650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 61), 'ext', False)
        keyword_90651 = ext_90650
        kwargs_90652 = {'ext': keyword_90651}
        # Getting the type of 'splev' (line 295)
        splev_90645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'splev', False)
        # Calling splev(args, kwargs) (line 295)
        splev_call_result_90653 = invoke(stypy.reporting.localization.Localization(__file__, 295, 38), splev_90645, *[list_90646, tck_90649], **kwargs_90652)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'ext' (line 295)
        ext_90654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 72), 'ext', False)
        # Getting the type of 'rstl' (line 295)
        rstl_90655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 67), 'rstl', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___90656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 67), rstl_90655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_90657 = invoke(stypy.reporting.localization.Localization(__file__, 295, 67), getitem___90656, ext_90654)
        
        # Processing the call keyword arguments (line 295)
        kwargs_90658 = {}
        # Getting the type of 'assert_array_almost_equal' (line 295)
        assert_array_almost_equal_90644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 295)
        assert_array_almost_equal_call_result_90659 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), assert_array_almost_equal_90644, *[splev_call_result_90653, subscript_call_result_90657], **kwargs_90658)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_raises(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'ValueError' (line 297)
        ValueError_90661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'ValueError', False)
        # Getting the type of 'splev' (line 297)
        splev_90662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 34), 'splev', False)
        
        # Obtaining an instance of the builtin type 'list' (line 297)
        list_90663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 297)
        # Adding element type (line 297)
        int_90664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 41), list_90663, int_90664)
        # Adding element type (line 297)
        int_90665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 297, 41), list_90663, int_90665)
        
        # Getting the type of 'tck' (line 297)
        tck_90666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 49), 'tck', False)
        # Processing the call keyword arguments (line 297)
        int_90667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 58), 'int')
        keyword_90668 = int_90667
        kwargs_90669 = {'ext': keyword_90668}
        # Getting the type of 'assert_raises' (line 297)
        assert_raises_90660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 297)
        assert_raises_call_result_90670 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assert_raises_90660, *[ValueError_90661, splev_90662, list_90663, tck_90666], **kwargs_90669)
        
        
        # ################# End of 'test_extrapolation_modes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_extrapolation_modes' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_90671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_extrapolation_modes'
        return stypy_return_type_90671


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplev.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSplev' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'TestSplev', TestSplev)
# Declaration of the 'TestSplder' class

class TestSplder(object, ):

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.setup_method.__dict__.__setitem__('stypy_function_name', 'TestSplder.setup_method')
        TestSplder.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.setup_method', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a BinOp to a Name (line 303):
        
        # Assigning a BinOp to a Name (line 303):
        
        # Call to linspace(...): (line 303)
        # Processing the call arguments (line 303)
        int_90674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 24), 'int')
        int_90675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 27), 'int')
        int_90676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 30), 'int')
        # Processing the call keyword arguments (line 303)
        kwargs_90677 = {}
        # Getting the type of 'np' (line 303)
        np_90672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 303)
        linspace_90673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 12), np_90672, 'linspace')
        # Calling linspace(args, kwargs) (line 303)
        linspace_call_result_90678 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), linspace_90673, *[int_90674, int_90675, int_90676], **kwargs_90677)
        
        int_90679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 36), 'int')
        # Applying the binary operator '**' (line 303)
        result_pow_90680 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 12), '**', linspace_call_result_90678, int_90679)
        
        # Assigning a type to the variable 'x' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'x', result_pow_90680)
        
        # Assigning a Call to a Name (line 304):
        
        # Assigning a Call to a Name (line 304):
        
        # Call to sin(...): (line 304)
        # Processing the call arguments (line 304)
        int_90683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 19), 'int')
        # Getting the type of 'x' (line 304)
        x_90684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'x', False)
        # Applying the binary operator '*' (line 304)
        result_mul_90685 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 19), '*', int_90683, x_90684)
        
        # Processing the call keyword arguments (line 304)
        kwargs_90686 = {}
        # Getting the type of 'np' (line 304)
        np_90681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'np', False)
        # Obtaining the member 'sin' of a type (line 304)
        sin_90682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), np_90681, 'sin')
        # Calling sin(args, kwargs) (line 304)
        sin_call_result_90687 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), sin_90682, *[result_mul_90685], **kwargs_90686)
        
        # Assigning a type to the variable 'y' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'y', sin_call_result_90687)
        
        # Assigning a Call to a Attribute (line 305):
        
        # Assigning a Call to a Attribute (line 305):
        
        # Call to splrep(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'x' (line 305)
        x_90689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 26), 'x', False)
        # Getting the type of 'y' (line 305)
        y_90690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'y', False)
        # Processing the call keyword arguments (line 305)
        kwargs_90691 = {}
        # Getting the type of 'splrep' (line 305)
        splrep_90688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'splrep', False)
        # Calling splrep(args, kwargs) (line 305)
        splrep_call_result_90692 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), splrep_90688, *[x_90689, y_90690], **kwargs_90691)
        
        # Getting the type of 'self' (line 305)
        self_90693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self')
        # Setting the type of the member 'spl' of a type (line 305)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_90693, 'spl', splrep_call_result_90692)
        
        # Call to assert_(...): (line 308)
        # Processing the call arguments (line 308)
        
        
        # Call to ptp(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_90705 = {}
        
        # Call to diff(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Obtaining the type of the subscript
        int_90697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 33), 'int')
        # Getting the type of 'self' (line 308)
        self_90698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'self', False)
        # Obtaining the member 'spl' of a type (line 308)
        spl_90699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), self_90698, 'spl')
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___90700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), spl_90699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_90701 = invoke(stypy.reporting.localization.Localization(__file__, 308, 24), getitem___90700, int_90697)
        
        # Processing the call keyword arguments (line 308)
        kwargs_90702 = {}
        # Getting the type of 'np' (line 308)
        np_90695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'np', False)
        # Obtaining the member 'diff' of a type (line 308)
        diff_90696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), np_90695, 'diff')
        # Calling diff(args, kwargs) (line 308)
        diff_call_result_90703 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), diff_90696, *[subscript_call_result_90701], **kwargs_90702)
        
        # Obtaining the member 'ptp' of a type (line 308)
        ptp_90704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), diff_call_result_90703, 'ptp')
        # Calling ptp(args, kwargs) (line 308)
        ptp_call_result_90706 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), ptp_90704, *[], **kwargs_90705)
        
        int_90707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 45), 'int')
        # Applying the binary operator '>' (line 308)
        result_gt_90708 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 16), '>', ptp_call_result_90706, int_90707)
        
        # Processing the call keyword arguments (line 308)
        kwargs_90709 = {}
        # Getting the type of 'assert_' (line 308)
        assert__90694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 308)
        assert__call_result_90710 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), assert__90694, *[result_gt_90708], **kwargs_90709)
        
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_90711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_90711


    @norecursion
    def test_inverse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inverse'
        module_type_store = module_type_store.open_function_context('test_inverse', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_inverse.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_inverse')
        TestSplder.test_inverse.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_inverse.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_inverse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_inverse', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to range(...): (line 312)
        # Processing the call arguments (line 312)
        int_90713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 23), 'int')
        # Processing the call keyword arguments (line 312)
        kwargs_90714 = {}
        # Getting the type of 'range' (line 312)
        range_90712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'range', False)
        # Calling range(args, kwargs) (line 312)
        range_call_result_90715 = invoke(stypy.reporting.localization.Localization(__file__, 312, 17), range_90712, *[int_90713], **kwargs_90714)
        
        # Testing the type of a for loop iterable (line 312)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 8), range_call_result_90715)
        # Getting the type of the for loop variable (line 312)
        for_loop_var_90716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 8), range_call_result_90715)
        # Assigning a type to the variable 'n' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'n', for_loop_var_90716)
        # SSA begins for a for statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to splantider(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'self' (line 313)
        self_90718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'self', False)
        # Obtaining the member 'spl' of a type (line 313)
        spl_90719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), self_90718, 'spl')
        # Getting the type of 'n' (line 313)
        n_90720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 40), 'n', False)
        # Processing the call keyword arguments (line 313)
        kwargs_90721 = {}
        # Getting the type of 'splantider' (line 313)
        splantider_90717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'splantider', False)
        # Calling splantider(args, kwargs) (line 313)
        splantider_call_result_90722 = invoke(stypy.reporting.localization.Localization(__file__, 313, 19), splantider_90717, *[spl_90719, n_90720], **kwargs_90721)
        
        # Assigning a type to the variable 'spl2' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'spl2', splantider_call_result_90722)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to splder(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'spl2' (line 314)
        spl2_90724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'spl2', False)
        # Getting the type of 'n' (line 314)
        n_90725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 32), 'n', False)
        # Processing the call keyword arguments (line 314)
        kwargs_90726 = {}
        # Getting the type of 'splder' (line 314)
        splder_90723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 19), 'splder', False)
        # Calling splder(args, kwargs) (line 314)
        splder_call_result_90727 = invoke(stypy.reporting.localization.Localization(__file__, 314, 19), splder_90723, *[spl2_90724, n_90725], **kwargs_90726)
        
        # Assigning a type to the variable 'spl3' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'spl3', splder_call_result_90727)
        
        # Call to assert_allclose(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Obtaining the type of the subscript
        int_90729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 37), 'int')
        # Getting the type of 'self' (line 315)
        self_90730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 28), 'self', False)
        # Obtaining the member 'spl' of a type (line 315)
        spl_90731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 28), self_90730, 'spl')
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___90732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 28), spl_90731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_90733 = invoke(stypy.reporting.localization.Localization(__file__, 315, 28), getitem___90732, int_90729)
        
        
        # Obtaining the type of the subscript
        int_90734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'int')
        # Getting the type of 'spl3' (line 315)
        spl3_90735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 41), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___90736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 41), spl3_90735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_90737 = invoke(stypy.reporting.localization.Localization(__file__, 315, 41), getitem___90736, int_90734)
        
        # Processing the call keyword arguments (line 315)
        kwargs_90738 = {}
        # Getting the type of 'assert_allclose' (line 315)
        assert_allclose_90728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 315)
        assert_allclose_call_result_90739 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), assert_allclose_90728, *[subscript_call_result_90733, subscript_call_result_90737], **kwargs_90738)
        
        
        # Call to assert_allclose(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining the type of the subscript
        int_90741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 37), 'int')
        # Getting the type of 'self' (line 316)
        self_90742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 28), 'self', False)
        # Obtaining the member 'spl' of a type (line 316)
        spl_90743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 28), self_90742, 'spl')
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___90744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 28), spl_90743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_90745 = invoke(stypy.reporting.localization.Localization(__file__, 316, 28), getitem___90744, int_90741)
        
        
        # Obtaining the type of the subscript
        int_90746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 46), 'int')
        # Getting the type of 'spl3' (line 316)
        spl3_90747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 41), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 316)
        getitem___90748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 41), spl3_90747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 316)
        subscript_call_result_90749 = invoke(stypy.reporting.localization.Localization(__file__, 316, 41), getitem___90748, int_90746)
        
        # Processing the call keyword arguments (line 316)
        kwargs_90750 = {}
        # Getting the type of 'assert_allclose' (line 316)
        assert_allclose_90740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 316)
        assert_allclose_call_result_90751 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), assert_allclose_90740, *[subscript_call_result_90745, subscript_call_result_90749], **kwargs_90750)
        
        
        # Call to assert_equal(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Obtaining the type of the subscript
        int_90753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 34), 'int')
        # Getting the type of 'self' (line 317)
        self_90754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 25), 'self', False)
        # Obtaining the member 'spl' of a type (line 317)
        spl_90755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 25), self_90754, 'spl')
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___90756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 25), spl_90755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_90757 = invoke(stypy.reporting.localization.Localization(__file__, 317, 25), getitem___90756, int_90753)
        
        
        # Obtaining the type of the subscript
        int_90758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 43), 'int')
        # Getting the type of 'spl3' (line 317)
        spl3_90759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 317)
        getitem___90760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 38), spl3_90759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 317)
        subscript_call_result_90761 = invoke(stypy.reporting.localization.Localization(__file__, 317, 38), getitem___90760, int_90758)
        
        # Processing the call keyword arguments (line 317)
        kwargs_90762 = {}
        # Getting the type of 'assert_equal' (line 317)
        assert_equal_90752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 317)
        assert_equal_call_result_90763 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), assert_equal_90752, *[subscript_call_result_90757, subscript_call_result_90761], **kwargs_90762)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_inverse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inverse' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_90764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inverse'
        return stypy_return_type_90764


    @norecursion
    def test_splder_vs_splev(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splder_vs_splev'
        module_type_store = module_type_store.open_function_context('test_splder_vs_splev', 319, 4, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_splder_vs_splev')
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_splder_vs_splev.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_splder_vs_splev', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splder_vs_splev', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splder_vs_splev(...)' code ##################

        
        
        # Call to range(...): (line 322)
        # Processing the call arguments (line 322)
        int_90766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 23), 'int')
        int_90767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'int')
        # Applying the binary operator '+' (line 322)
        result_add_90768 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 23), '+', int_90766, int_90767)
        
        # Processing the call keyword arguments (line 322)
        kwargs_90769 = {}
        # Getting the type of 'range' (line 322)
        range_90765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'range', False)
        # Calling range(args, kwargs) (line 322)
        range_call_result_90770 = invoke(stypy.reporting.localization.Localization(__file__, 322, 17), range_90765, *[result_add_90768], **kwargs_90769)
        
        # Testing the type of a for loop iterable (line 322)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 8), range_call_result_90770)
        # Getting the type of the for loop variable (line 322)
        for_loop_var_90771 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 8), range_call_result_90770)
        # Assigning a type to the variable 'n' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'n', for_loop_var_90771)
        # SSA begins for a for statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to linspace(...): (line 324)
        # Processing the call arguments (line 324)
        int_90774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 29), 'int')
        int_90775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 33), 'int')
        int_90776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 36), 'int')
        # Processing the call keyword arguments (line 324)
        kwargs_90777 = {}
        # Getting the type of 'np' (line 324)
        np_90772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 17), 'np', False)
        # Obtaining the member 'linspace' of a type (line 324)
        linspace_90773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 17), np_90772, 'linspace')
        # Calling linspace(args, kwargs) (line 324)
        linspace_call_result_90778 = invoke(stypy.reporting.localization.Localization(__file__, 324, 17), linspace_90773, *[int_90774, int_90775, int_90776], **kwargs_90777)
        
        # Assigning a type to the variable 'xx' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'xx', linspace_call_result_90778)
        
        
        # Getting the type of 'n' (line 325)
        n_90779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'n')
        int_90780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 20), 'int')
        # Applying the binary operator '==' (line 325)
        result_eq_90781 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 15), '==', n_90779, int_90780)
        
        # Testing the type of an if condition (line 325)
        if_condition_90782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 325, 12), result_eq_90781)
        # Assigning a type to the variable 'if_condition_90782' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'if_condition_90782', if_condition_90782)
        # SSA begins for if statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 328):
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'xx' (line 328)
        xx_90783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'xx')
        int_90784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 31), 'int')
        # Applying the binary operator '>=' (line 328)
        result_ge_90785 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 25), '>=', xx_90783, int_90784)
        
        
        # Getting the type of 'xx' (line 328)
        xx_90786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 37), 'xx')
        int_90787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 43), 'int')
        # Applying the binary operator '<=' (line 328)
        result_le_90788 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 37), '<=', xx_90786, int_90787)
        
        # Applying the binary operator '&' (line 328)
        result_and__90789 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 24), '&', result_ge_90785, result_le_90788)
        
        # Getting the type of 'xx' (line 328)
        xx_90790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'xx')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___90791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), xx_90790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_90792 = invoke(stypy.reporting.localization.Localization(__file__, 328, 21), getitem___90791, result_and__90789)
        
        # Assigning a type to the variable 'xx' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'xx', subscript_call_result_90792)
        # SSA join for if statement (line 325)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 330):
        
        # Assigning a Call to a Name (line 330):
        
        # Call to splev(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'xx' (line 330)
        xx_90794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 'xx', False)
        # Getting the type of 'self' (line 330)
        self_90795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 27), 'self', False)
        # Obtaining the member 'spl' of a type (line 330)
        spl_90796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 27), self_90795, 'spl')
        # Getting the type of 'n' (line 330)
        n_90797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 37), 'n', False)
        # Processing the call keyword arguments (line 330)
        kwargs_90798 = {}
        # Getting the type of 'splev' (line 330)
        splev_90793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 17), 'splev', False)
        # Calling splev(args, kwargs) (line 330)
        splev_call_result_90799 = invoke(stypy.reporting.localization.Localization(__file__, 330, 17), splev_90793, *[xx_90794, spl_90796, n_90797], **kwargs_90798)
        
        # Assigning a type to the variable 'dy' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'dy', splev_call_result_90799)
        
        # Assigning a Call to a Name (line 331):
        
        # Assigning a Call to a Name (line 331):
        
        # Call to splder(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'self' (line 331)
        self_90801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'self', False)
        # Obtaining the member 'spl' of a type (line 331)
        spl_90802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 26), self_90801, 'spl')
        # Getting the type of 'n' (line 331)
        n_90803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'n', False)
        # Processing the call keyword arguments (line 331)
        kwargs_90804 = {}
        # Getting the type of 'splder' (line 331)
        splder_90800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'splder', False)
        # Calling splder(args, kwargs) (line 331)
        splder_call_result_90805 = invoke(stypy.reporting.localization.Localization(__file__, 331, 19), splder_90800, *[spl_90802, n_90803], **kwargs_90804)
        
        # Assigning a type to the variable 'spl2' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'spl2', splder_call_result_90805)
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to splev(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'xx' (line 332)
        xx_90807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 24), 'xx', False)
        # Getting the type of 'spl2' (line 332)
        spl2_90808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'spl2', False)
        # Processing the call keyword arguments (line 332)
        kwargs_90809 = {}
        # Getting the type of 'splev' (line 332)
        splev_90806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'splev', False)
        # Calling splev(args, kwargs) (line 332)
        splev_call_result_90810 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), splev_90806, *[xx_90807, spl2_90808], **kwargs_90809)
        
        # Assigning a type to the variable 'dy2' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'dy2', splev_call_result_90810)
        
        
        # Getting the type of 'n' (line 333)
        n_90811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'n')
        int_90812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 20), 'int')
        # Applying the binary operator '==' (line 333)
        result_eq_90813 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 15), '==', n_90811, int_90812)
        
        # Testing the type of an if condition (line 333)
        if_condition_90814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 12), result_eq_90813)
        # Assigning a type to the variable 'if_condition_90814' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'if_condition_90814', if_condition_90814)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to assert_allclose(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'dy' (line 334)
        dy_90816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'dy', False)
        # Getting the type of 'dy2' (line 334)
        dy2_90817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 36), 'dy2', False)
        # Processing the call keyword arguments (line 334)
        float_90818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 46), 'float')
        keyword_90819 = float_90818
        kwargs_90820 = {'rtol': keyword_90819}
        # Getting the type of 'assert_allclose' (line 334)
        assert_allclose_90815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 334)
        assert_allclose_call_result_90821 = invoke(stypy.reporting.localization.Localization(__file__, 334, 16), assert_allclose_90815, *[dy_90816, dy2_90817], **kwargs_90820)
        
        # SSA branch for the else part of an if statement (line 333)
        module_type_store.open_ssa_branch('else')
        
        # Call to assert_allclose(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'dy' (line 336)
        dy_90823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 32), 'dy', False)
        # Getting the type of 'dy2' (line 336)
        dy2_90824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 36), 'dy2', False)
        # Processing the call keyword arguments (line 336)
        kwargs_90825 = {}
        # Getting the type of 'assert_allclose' (line 336)
        assert_allclose_90822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 336)
        assert_allclose_call_result_90826 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), assert_allclose_90822, *[dy_90823, dy2_90824], **kwargs_90825)
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_splder_vs_splev(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splder_vs_splev' in the type store
        # Getting the type of 'stypy_return_type' (line 319)
        stypy_return_type_90827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90827)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splder_vs_splev'
        return stypy_return_type_90827


    @norecursion
    def test_splantider_vs_splint(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_splantider_vs_splint'
        module_type_store = module_type_store.open_function_context('test_splantider_vs_splint', 338, 4, False)
        # Assigning a type to the variable 'self' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_splantider_vs_splint')
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_splantider_vs_splint.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_splantider_vs_splint', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_splantider_vs_splint', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_splantider_vs_splint(...)' code ##################

        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to splantider(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'self' (line 340)
        self_90829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 26), 'self', False)
        # Obtaining the member 'spl' of a type (line 340)
        spl_90830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 26), self_90829, 'spl')
        # Processing the call keyword arguments (line 340)
        kwargs_90831 = {}
        # Getting the type of 'splantider' (line 340)
        splantider_90828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'splantider', False)
        # Calling splantider(args, kwargs) (line 340)
        splantider_call_result_90832 = invoke(stypy.reporting.localization.Localization(__file__, 340, 15), splantider_90828, *[spl_90830], **kwargs_90831)
        
        # Assigning a type to the variable 'spl2' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'spl2', splantider_call_result_90832)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to linspace(...): (line 344)
        # Processing the call arguments (line 344)
        int_90835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 25), 'int')
        int_90836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 28), 'int')
        int_90837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 31), 'int')
        # Processing the call keyword arguments (line 344)
        kwargs_90838 = {}
        # Getting the type of 'np' (line 344)
        np_90833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'np', False)
        # Obtaining the member 'linspace' of a type (line 344)
        linspace_90834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 13), np_90833, 'linspace')
        # Calling linspace(args, kwargs) (line 344)
        linspace_call_result_90839 = invoke(stypy.reporting.localization.Localization(__file__, 344, 13), linspace_90834, *[int_90835, int_90836, int_90837], **kwargs_90838)
        
        # Assigning a type to the variable 'xx' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'xx', linspace_call_result_90839)
        
        # Getting the type of 'xx' (line 346)
        xx_90840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 18), 'xx')
        # Testing the type of a for loop iterable (line 346)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 346, 8), xx_90840)
        # Getting the type of the for loop variable (line 346)
        for_loop_var_90841 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 346, 8), xx_90840)
        # Assigning a type to the variable 'x1' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'x1', for_loop_var_90841)
        # SSA begins for a for statement (line 346)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'xx' (line 347)
        xx_90842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'xx')
        # Testing the type of a for loop iterable (line 347)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 347, 12), xx_90842)
        # Getting the type of the for loop variable (line 347)
        for_loop_var_90843 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 347, 12), xx_90842)
        # Assigning a type to the variable 'x2' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'x2', for_loop_var_90843)
        # SSA begins for a for statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to splint(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'x1' (line 348)
        x1_90845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 28), 'x1', False)
        # Getting the type of 'x2' (line 348)
        x2_90846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 32), 'x2', False)
        # Getting the type of 'self' (line 348)
        self_90847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 36), 'self', False)
        # Obtaining the member 'spl' of a type (line 348)
        spl_90848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 36), self_90847, 'spl')
        # Processing the call keyword arguments (line 348)
        kwargs_90849 = {}
        # Getting the type of 'splint' (line 348)
        splint_90844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'splint', False)
        # Calling splint(args, kwargs) (line 348)
        splint_call_result_90850 = invoke(stypy.reporting.localization.Localization(__file__, 348, 21), splint_90844, *[x1_90845, x2_90846, spl_90848], **kwargs_90849)
        
        # Assigning a type to the variable 'y1' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'y1', splint_call_result_90850)
        
        # Assigning a BinOp to a Name (line 349):
        
        # Assigning a BinOp to a Name (line 349):
        
        # Call to splev(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'x2' (line 349)
        x2_90852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'x2', False)
        # Getting the type of 'spl2' (line 349)
        spl2_90853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 31), 'spl2', False)
        # Processing the call keyword arguments (line 349)
        kwargs_90854 = {}
        # Getting the type of 'splev' (line 349)
        splev_90851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'splev', False)
        # Calling splev(args, kwargs) (line 349)
        splev_call_result_90855 = invoke(stypy.reporting.localization.Localization(__file__, 349, 21), splev_90851, *[x2_90852, spl2_90853], **kwargs_90854)
        
        
        # Call to splev(...): (line 349)
        # Processing the call arguments (line 349)
        # Getting the type of 'x1' (line 349)
        x1_90857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'x1', False)
        # Getting the type of 'spl2' (line 349)
        spl2_90858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 49), 'spl2', False)
        # Processing the call keyword arguments (line 349)
        kwargs_90859 = {}
        # Getting the type of 'splev' (line 349)
        splev_90856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 39), 'splev', False)
        # Calling splev(args, kwargs) (line 349)
        splev_call_result_90860 = invoke(stypy.reporting.localization.Localization(__file__, 349, 39), splev_90856, *[x1_90857, spl2_90858], **kwargs_90859)
        
        # Applying the binary operator '-' (line 349)
        result_sub_90861 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 21), '-', splev_call_result_90855, splev_call_result_90860)
        
        # Assigning a type to the variable 'y2' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'y2', result_sub_90861)
        
        # Call to assert_allclose(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'y1' (line 350)
        y1_90863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 32), 'y1', False)
        # Getting the type of 'y2' (line 350)
        y2_90864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), 'y2', False)
        # Processing the call keyword arguments (line 350)
        kwargs_90865 = {}
        # Getting the type of 'assert_allclose' (line 350)
        assert_allclose_90862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 350)
        assert_allclose_call_result_90866 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), assert_allclose_90862, *[y1_90863, y2_90864], **kwargs_90865)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_splantider_vs_splint(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_splantider_vs_splint' in the type store
        # Getting the type of 'stypy_return_type' (line 338)
        stypy_return_type_90867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90867)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_splantider_vs_splint'
        return stypy_return_type_90867


    @norecursion
    def test_order0_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_order0_diff'
        module_type_store = module_type_store.open_function_context('test_order0_diff', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_order0_diff')
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_order0_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_order0_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_order0_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_order0_diff(...)' code ##################

        
        # Call to assert_raises(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'ValueError' (line 353)
        ValueError_90869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 22), 'ValueError', False)
        # Getting the type of 'splder' (line 353)
        splder_90870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 34), 'splder', False)
        # Getting the type of 'self' (line 353)
        self_90871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 42), 'self', False)
        # Obtaining the member 'spl' of a type (line 353)
        spl_90872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 42), self_90871, 'spl')
        int_90873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 52), 'int')
        # Processing the call keyword arguments (line 353)
        kwargs_90874 = {}
        # Getting the type of 'assert_raises' (line 353)
        assert_raises_90868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 353)
        assert_raises_call_result_90875 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert_raises_90868, *[ValueError_90869, splder_90870, spl_90872, int_90873], **kwargs_90874)
        
        
        # ################# End of 'test_order0_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_order0_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_90876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_order0_diff'
        return stypy_return_type_90876


    @norecursion
    def test_kink(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kink'
        module_type_store = module_type_store.open_function_context('test_kink', 355, 4, False)
        # Assigning a type to the variable 'self' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_kink.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_kink.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_kink.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_kink.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_kink')
        TestSplder.test_kink.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_kink.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_kink.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_kink.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_kink.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_kink.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_kink.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_kink', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kink', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kink(...)' code ##################

        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to insert(...): (line 358)
        # Processing the call arguments (line 358)
        float_90878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 22), 'float')
        # Getting the type of 'self' (line 358)
        self_90879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'self', False)
        # Obtaining the member 'spl' of a type (line 358)
        spl_90880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 27), self_90879, 'spl')
        # Processing the call keyword arguments (line 358)
        int_90881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 39), 'int')
        keyword_90882 = int_90881
        kwargs_90883 = {'m': keyword_90882}
        # Getting the type of 'insert' (line 358)
        insert_90877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'insert', False)
        # Calling insert(args, kwargs) (line 358)
        insert_call_result_90884 = invoke(stypy.reporting.localization.Localization(__file__, 358, 15), insert_90877, *[float_90878, spl_90880], **kwargs_90883)
        
        # Assigning a type to the variable 'spl2' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'spl2', insert_call_result_90884)
        
        # Call to splder(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'spl2' (line 359)
        spl2_90886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'spl2', False)
        int_90887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 21), 'int')
        # Processing the call keyword arguments (line 359)
        kwargs_90888 = {}
        # Getting the type of 'splder' (line 359)
        splder_90885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'splder', False)
        # Calling splder(args, kwargs) (line 359)
        splder_call_result_90889 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), splder_90885, *[spl2_90886, int_90887], **kwargs_90888)
        
        
        # Call to assert_raises(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'ValueError' (line 360)
        ValueError_90891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'ValueError', False)
        # Getting the type of 'splder' (line 360)
        splder_90892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'splder', False)
        # Getting the type of 'spl2' (line 360)
        spl2_90893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 42), 'spl2', False)
        int_90894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 48), 'int')
        # Processing the call keyword arguments (line 360)
        kwargs_90895 = {}
        # Getting the type of 'assert_raises' (line 360)
        assert_raises_90890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 360)
        assert_raises_call_result_90896 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assert_raises_90890, *[ValueError_90891, splder_90892, spl2_90893, int_90894], **kwargs_90895)
        
        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to insert(...): (line 362)
        # Processing the call arguments (line 362)
        float_90898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 22), 'float')
        # Getting the type of 'self' (line 362)
        self_90899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'self', False)
        # Obtaining the member 'spl' of a type (line 362)
        spl_90900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 27), self_90899, 'spl')
        # Processing the call keyword arguments (line 362)
        int_90901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 39), 'int')
        keyword_90902 = int_90901
        kwargs_90903 = {'m': keyword_90902}
        # Getting the type of 'insert' (line 362)
        insert_90897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'insert', False)
        # Calling insert(args, kwargs) (line 362)
        insert_call_result_90904 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), insert_90897, *[float_90898, spl_90900], **kwargs_90903)
        
        # Assigning a type to the variable 'spl2' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'spl2', insert_call_result_90904)
        
        # Call to splder(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'spl2' (line 363)
        spl2_90906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'spl2', False)
        int_90907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 21), 'int')
        # Processing the call keyword arguments (line 363)
        kwargs_90908 = {}
        # Getting the type of 'splder' (line 363)
        splder_90905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'splder', False)
        # Calling splder(args, kwargs) (line 363)
        splder_call_result_90909 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), splder_90905, *[spl2_90906, int_90907], **kwargs_90908)
        
        
        # Call to assert_raises(...): (line 364)
        # Processing the call arguments (line 364)
        # Getting the type of 'ValueError' (line 364)
        ValueError_90911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 22), 'ValueError', False)
        # Getting the type of 'splder' (line 364)
        splder_90912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 34), 'splder', False)
        # Getting the type of 'spl2' (line 364)
        spl2_90913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 42), 'spl2', False)
        int_90914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 48), 'int')
        # Processing the call keyword arguments (line 364)
        kwargs_90915 = {}
        # Getting the type of 'assert_raises' (line 364)
        assert_raises_90910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 364)
        assert_raises_call_result_90916 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), assert_raises_90910, *[ValueError_90911, splder_90912, spl2_90913, int_90914], **kwargs_90915)
        
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to insert(...): (line 366)
        # Processing the call arguments (line 366)
        float_90918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 22), 'float')
        # Getting the type of 'self' (line 366)
        self_90919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 27), 'self', False)
        # Obtaining the member 'spl' of a type (line 366)
        spl_90920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 27), self_90919, 'spl')
        # Processing the call keyword arguments (line 366)
        int_90921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 39), 'int')
        keyword_90922 = int_90921
        kwargs_90923 = {'m': keyword_90922}
        # Getting the type of 'insert' (line 366)
        insert_90917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 15), 'insert', False)
        # Calling insert(args, kwargs) (line 366)
        insert_call_result_90924 = invoke(stypy.reporting.localization.Localization(__file__, 366, 15), insert_90917, *[float_90918, spl_90920], **kwargs_90923)
        
        # Assigning a type to the variable 'spl2' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'spl2', insert_call_result_90924)
        
        # Call to assert_raises(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'ValueError' (line 367)
        ValueError_90926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'ValueError', False)
        # Getting the type of 'splder' (line 367)
        splder_90927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'splder', False)
        # Getting the type of 'spl2' (line 367)
        spl2_90928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 42), 'spl2', False)
        int_90929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 48), 'int')
        # Processing the call keyword arguments (line 367)
        kwargs_90930 = {}
        # Getting the type of 'assert_raises' (line 367)
        assert_raises_90925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 367)
        assert_raises_call_result_90931 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), assert_raises_90925, *[ValueError_90926, splder_90927, spl2_90928, int_90929], **kwargs_90930)
        
        
        # ################# End of 'test_kink(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kink' in the type store
        # Getting the type of 'stypy_return_type' (line 355)
        stypy_return_type_90932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_90932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kink'
        return stypy_return_type_90932


    @norecursion
    def test_multidim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multidim'
        module_type_store = module_type_store.open_function_context('test_multidim', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSplder.test_multidim.__dict__.__setitem__('stypy_localization', localization)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_function_name', 'TestSplder.test_multidim')
        TestSplder.test_multidim.__dict__.__setitem__('stypy_param_names_list', [])
        TestSplder.test_multidim.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSplder.test_multidim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.test_multidim', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multidim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multidim(...)' code ##################

        
        
        # Call to range(...): (line 371)
        # Processing the call arguments (line 371)
        int_90934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 23), 'int')
        # Processing the call keyword arguments (line 371)
        kwargs_90935 = {}
        # Getting the type of 'range' (line 371)
        range_90933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'range', False)
        # Calling range(args, kwargs) (line 371)
        range_call_result_90936 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), range_90933, *[int_90934], **kwargs_90935)
        
        # Testing the type of a for loop iterable (line 371)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 371, 8), range_call_result_90936)
        # Getting the type of the for loop variable (line 371)
        for_loop_var_90937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 371, 8), range_call_result_90936)
        # Assigning a type to the variable 'n' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'n', for_loop_var_90937)
        # SSA begins for a for statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Tuple (line 372):
        
        # Assigning a Subscript to a Name (line 372):
        
        # Obtaining the type of the subscript
        int_90938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'int')
        # Getting the type of 'self' (line 372)
        self_90939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'self')
        # Obtaining the member 'spl' of a type (line 372)
        spl_90940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 22), self_90939, 'spl')
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___90941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), spl_90940, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_90942 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), getitem___90941, int_90938)
        
        # Assigning a type to the variable 'tuple_var_assignment_88956' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88956', subscript_call_result_90942)
        
        # Assigning a Subscript to a Name (line 372):
        
        # Obtaining the type of the subscript
        int_90943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'int')
        # Getting the type of 'self' (line 372)
        self_90944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'self')
        # Obtaining the member 'spl' of a type (line 372)
        spl_90945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 22), self_90944, 'spl')
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___90946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), spl_90945, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_90947 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), getitem___90946, int_90943)
        
        # Assigning a type to the variable 'tuple_var_assignment_88957' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88957', subscript_call_result_90947)
        
        # Assigning a Subscript to a Name (line 372):
        
        # Obtaining the type of the subscript
        int_90948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 12), 'int')
        # Getting the type of 'self' (line 372)
        self_90949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'self')
        # Obtaining the member 'spl' of a type (line 372)
        spl_90950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 22), self_90949, 'spl')
        # Obtaining the member '__getitem__' of a type (line 372)
        getitem___90951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), spl_90950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 372)
        subscript_call_result_90952 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), getitem___90951, int_90948)
        
        # Assigning a type to the variable 'tuple_var_assignment_88958' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88958', subscript_call_result_90952)
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'tuple_var_assignment_88956' (line 372)
        tuple_var_assignment_88956_90953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88956')
        # Assigning a type to the variable 't' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 't', tuple_var_assignment_88956_90953)
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'tuple_var_assignment_88957' (line 372)
        tuple_var_assignment_88957_90954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88957')
        # Assigning a type to the variable 'c' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'c', tuple_var_assignment_88957_90954)
        
        # Assigning a Name to a Name (line 372):
        # Getting the type of 'tuple_var_assignment_88958' (line 372)
        tuple_var_assignment_88958_90955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'tuple_var_assignment_88958')
        # Assigning a type to the variable 'k' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 18), 'k', tuple_var_assignment_88958_90955)
        
        # Assigning a Subscript to a Name (line 373):
        
        # Assigning a Subscript to a Name (line 373):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 373)
        tuple_90956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 373)
        # Adding element type (line 373)
        # Getting the type of 'c' (line 373)
        c_90957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 23), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), tuple_90956, c_90957)
        # Adding element type (line 373)
        # Getting the type of 'c' (line 373)
        c_90958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), tuple_90956, c_90958)
        # Adding element type (line 373)
        # Getting the type of 'c' (line 373)
        c_90959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 29), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), tuple_90956, c_90959)
        
        # Getting the type of 'np' (line 373)
        np_90960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'np')
        # Obtaining the member 'c_' of a type (line 373)
        c__90961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 17), np_90960, 'c_')
        # Obtaining the member '__getitem__' of a type (line 373)
        getitem___90962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 17), c__90961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 373)
        subscript_call_result_90963 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), getitem___90962, tuple_90956)
        
        # Assigning a type to the variable 'c2' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'c2', subscript_call_result_90963)
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to dstack(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_90966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        # Getting the type of 'c2' (line 374)
        c2_90967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'c2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 28), tuple_90966, c2_90967)
        # Adding element type (line 374)
        # Getting the type of 'c2' (line 374)
        c2_90968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 32), 'c2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 28), tuple_90966, c2_90968)
        
        # Processing the call keyword arguments (line 374)
        kwargs_90969 = {}
        # Getting the type of 'np' (line 374)
        np_90964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 17), 'np', False)
        # Obtaining the member 'dstack' of a type (line 374)
        dstack_90965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 17), np_90964, 'dstack')
        # Calling dstack(args, kwargs) (line 374)
        dstack_call_result_90970 = invoke(stypy.reporting.localization.Localization(__file__, 374, 17), dstack_90965, *[tuple_90966], **kwargs_90969)
        
        # Assigning a type to the variable 'c2' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'c2', dstack_call_result_90970)
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Call to splantider(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_90972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        # Getting the type of 't' (line 376)
        t_90973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 31), 't', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 31), tuple_90972, t_90973)
        # Adding element type (line 376)
        # Getting the type of 'c2' (line 376)
        c2_90974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 34), 'c2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 31), tuple_90972, c2_90974)
        # Adding element type (line 376)
        # Getting the type of 'k' (line 376)
        k_90975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 38), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 31), tuple_90972, k_90975)
        
        # Getting the type of 'n' (line 376)
        n_90976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 42), 'n', False)
        # Processing the call keyword arguments (line 376)
        kwargs_90977 = {}
        # Getting the type of 'splantider' (line 376)
        splantider_90971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'splantider', False)
        # Calling splantider(args, kwargs) (line 376)
        splantider_call_result_90978 = invoke(stypy.reporting.localization.Localization(__file__, 376, 19), splantider_90971, *[tuple_90972, n_90976], **kwargs_90977)
        
        # Assigning a type to the variable 'spl2' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'spl2', splantider_call_result_90978)
        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to splder(...): (line 377)
        # Processing the call arguments (line 377)
        # Getting the type of 'spl2' (line 377)
        spl2_90980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 26), 'spl2', False)
        # Getting the type of 'n' (line 377)
        n_90981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 32), 'n', False)
        # Processing the call keyword arguments (line 377)
        kwargs_90982 = {}
        # Getting the type of 'splder' (line 377)
        splder_90979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'splder', False)
        # Calling splder(args, kwargs) (line 377)
        splder_call_result_90983 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), splder_90979, *[spl2_90980, n_90981], **kwargs_90982)
        
        # Assigning a type to the variable 'spl3' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'spl3', splder_call_result_90983)
        
        # Call to assert_allclose(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 't' (line 379)
        t_90985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 28), 't', False)
        
        # Obtaining the type of the subscript
        int_90986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 36), 'int')
        # Getting the type of 'spl3' (line 379)
        spl3_90987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 31), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___90988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 31), spl3_90987, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 379)
        subscript_call_result_90989 = invoke(stypy.reporting.localization.Localization(__file__, 379, 31), getitem___90988, int_90986)
        
        # Processing the call keyword arguments (line 379)
        kwargs_90990 = {}
        # Getting the type of 'assert_allclose' (line 379)
        assert_allclose_90984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 379)
        assert_allclose_call_result_90991 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), assert_allclose_90984, *[t_90985, subscript_call_result_90989], **kwargs_90990)
        
        
        # Call to assert_allclose(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'c2' (line 380)
        c2_90993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 28), 'c2', False)
        
        # Obtaining the type of the subscript
        int_90994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 37), 'int')
        # Getting the type of 'spl3' (line 380)
        spl3_90995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 32), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___90996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 32), spl3_90995, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 380)
        subscript_call_result_90997 = invoke(stypy.reporting.localization.Localization(__file__, 380, 32), getitem___90996, int_90994)
        
        # Processing the call keyword arguments (line 380)
        kwargs_90998 = {}
        # Getting the type of 'assert_allclose' (line 380)
        assert_allclose_90992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 380)
        assert_allclose_call_result_90999 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), assert_allclose_90992, *[c2_90993, subscript_call_result_90997], **kwargs_90998)
        
        
        # Call to assert_equal(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'k' (line 381)
        k_91001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'k', False)
        
        # Obtaining the type of the subscript
        int_91002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 33), 'int')
        # Getting the type of 'spl3' (line 381)
        spl3_91003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 28), 'spl3', False)
        # Obtaining the member '__getitem__' of a type (line 381)
        getitem___91004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 28), spl3_91003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 381)
        subscript_call_result_91005 = invoke(stypy.reporting.localization.Localization(__file__, 381, 28), getitem___91004, int_91002)
        
        # Processing the call keyword arguments (line 381)
        kwargs_91006 = {}
        # Getting the type of 'assert_equal' (line 381)
        assert_equal_91000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 381)
        assert_equal_call_result_91007 = invoke(stypy.reporting.localization.Localization(__file__, 381, 12), assert_equal_91000, *[k_91001, subscript_call_result_91005], **kwargs_91006)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_multidim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multidim' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_91008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multidim'
        return stypy_return_type_91008


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 300, 0, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSplder.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSplder' (line 300)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 0), 'TestSplder', TestSplder)
# Declaration of the 'TestBisplrep' class

class TestBisplrep(object, ):

    @norecursion
    def test_overflow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_overflow'
        module_type_store = module_type_store.open_function_context('test_overflow', 385, 4, False)
        # Assigning a type to the variable 'self' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_localization', localization)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_function_name', 'TestBisplrep.test_overflow')
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_param_names_list', [])
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBisplrep.test_overflow.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBisplrep.test_overflow', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_overflow', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_overflow(...)' code ##################

        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to linspace(...): (line 386)
        # Processing the call arguments (line 386)
        int_91011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 24), 'int')
        int_91012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 27), 'int')
        int_91013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 30), 'int')
        # Processing the call keyword arguments (line 386)
        kwargs_91014 = {}
        # Getting the type of 'np' (line 386)
        np_91009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 386)
        linspace_91010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), np_91009, 'linspace')
        # Calling linspace(args, kwargs) (line 386)
        linspace_call_result_91015 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), linspace_91010, *[int_91011, int_91012, int_91013], **kwargs_91014)
        
        # Assigning a type to the variable 'a' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'a', linspace_call_result_91015)
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to linspace(...): (line 387)
        # Processing the call arguments (line 387)
        int_91018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 24), 'int')
        int_91019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 27), 'int')
        int_91020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 30), 'int')
        # Processing the call keyword arguments (line 387)
        kwargs_91021 = {}
        # Getting the type of 'np' (line 387)
        np_91016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 387)
        linspace_91017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 12), np_91016, 'linspace')
        # Calling linspace(args, kwargs) (line 387)
        linspace_call_result_91022 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), linspace_91017, *[int_91018, int_91019, int_91020], **kwargs_91021)
        
        # Assigning a type to the variable 'b' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'b', linspace_call_result_91022)
        
        # Assigning a Call to a Tuple (line 388):
        
        # Assigning a Subscript to a Name (line 388):
        
        # Obtaining the type of the subscript
        int_91023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 8), 'int')
        
        # Call to meshgrid(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'a' (line 388)
        a_91026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 27), 'a', False)
        # Getting the type of 'b' (line 388)
        b_91027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 30), 'b', False)
        # Processing the call keyword arguments (line 388)
        kwargs_91028 = {}
        # Getting the type of 'np' (line 388)
        np_91024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'np', False)
        # Obtaining the member 'meshgrid' of a type (line 388)
        meshgrid_91025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), np_91024, 'meshgrid')
        # Calling meshgrid(args, kwargs) (line 388)
        meshgrid_call_result_91029 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), meshgrid_91025, *[a_91026, b_91027], **kwargs_91028)
        
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___91030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), meshgrid_call_result_91029, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_91031 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), getitem___91030, int_91023)
        
        # Assigning a type to the variable 'tuple_var_assignment_88959' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_88959', subscript_call_result_91031)
        
        # Assigning a Subscript to a Name (line 388):
        
        # Obtaining the type of the subscript
        int_91032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 8), 'int')
        
        # Call to meshgrid(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'a' (line 388)
        a_91035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 27), 'a', False)
        # Getting the type of 'b' (line 388)
        b_91036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 30), 'b', False)
        # Processing the call keyword arguments (line 388)
        kwargs_91037 = {}
        # Getting the type of 'np' (line 388)
        np_91033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'np', False)
        # Obtaining the member 'meshgrid' of a type (line 388)
        meshgrid_91034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), np_91033, 'meshgrid')
        # Calling meshgrid(args, kwargs) (line 388)
        meshgrid_call_result_91038 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), meshgrid_91034, *[a_91035, b_91036], **kwargs_91037)
        
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___91039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), meshgrid_call_result_91038, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_91040 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), getitem___91039, int_91032)
        
        # Assigning a type to the variable 'tuple_var_assignment_88960' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_88960', subscript_call_result_91040)
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'tuple_var_assignment_88959' (line 388)
        tuple_var_assignment_88959_91041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_88959')
        # Assigning a type to the variable 'x' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'x', tuple_var_assignment_88959_91041)
        
        # Assigning a Name to a Name (line 388):
        # Getting the type of 'tuple_var_assignment_88960' (line 388)
        tuple_var_assignment_88960_91042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'tuple_var_assignment_88960')
        # Assigning a type to the variable 'y' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'y', tuple_var_assignment_88960_91042)
        
        # Assigning a Call to a Name (line 389):
        
        # Assigning a Call to a Name (line 389):
        
        # Call to rand(...): (line 389)
        # Getting the type of 'x' (line 389)
        x_91046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'x', False)
        # Obtaining the member 'shape' of a type (line 389)
        shape_91047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 28), x_91046, 'shape')
        # Processing the call keyword arguments (line 389)
        kwargs_91048 = {}
        # Getting the type of 'np' (line 389)
        np_91043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 389)
        random_91044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), np_91043, 'random')
        # Obtaining the member 'rand' of a type (line 389)
        rand_91045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), random_91044, 'rand')
        # Calling rand(args, kwargs) (line 389)
        rand_call_result_91049 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), rand_91045, *[shape_91047], **kwargs_91048)
        
        # Assigning a type to the variable 'z' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'z', rand_call_result_91049)
        
        # Call to assert_raises(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'OverflowError' (line 390)
        OverflowError_91051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'OverflowError', False)
        # Getting the type of 'bisplrep' (line 390)
        bisplrep_91052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 37), 'bisplrep', False)
        
        # Call to ravel(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_91055 = {}
        # Getting the type of 'x' (line 390)
        x_91053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 47), 'x', False)
        # Obtaining the member 'ravel' of a type (line 390)
        ravel_91054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 47), x_91053, 'ravel')
        # Calling ravel(args, kwargs) (line 390)
        ravel_call_result_91056 = invoke(stypy.reporting.localization.Localization(__file__, 390, 47), ravel_91054, *[], **kwargs_91055)
        
        
        # Call to ravel(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_91059 = {}
        # Getting the type of 'y' (line 390)
        y_91057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 58), 'y', False)
        # Obtaining the member 'ravel' of a type (line 390)
        ravel_91058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 58), y_91057, 'ravel')
        # Calling ravel(args, kwargs) (line 390)
        ravel_call_result_91060 = invoke(stypy.reporting.localization.Localization(__file__, 390, 58), ravel_91058, *[], **kwargs_91059)
        
        
        # Call to ravel(...): (line 390)
        # Processing the call keyword arguments (line 390)
        kwargs_91063 = {}
        # Getting the type of 'z' (line 390)
        z_91061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 69), 'z', False)
        # Obtaining the member 'ravel' of a type (line 390)
        ravel_91062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 69), z_91061, 'ravel')
        # Calling ravel(args, kwargs) (line 390)
        ravel_call_result_91064 = invoke(stypy.reporting.localization.Localization(__file__, 390, 69), ravel_91062, *[], **kwargs_91063)
        
        # Processing the call keyword arguments (line 390)
        int_91065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 82), 'int')
        keyword_91066 = int_91065
        kwargs_91067 = {'s': keyword_91066}
        # Getting the type of 'assert_raises' (line 390)
        assert_raises_91050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 390)
        assert_raises_call_result_91068 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), assert_raises_91050, *[OverflowError_91051, bisplrep_91052, ravel_call_result_91056, ravel_call_result_91060, ravel_call_result_91064], **kwargs_91067)
        
        
        # ################# End of 'test_overflow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_overflow' in the type store
        # Getting the type of 'stypy_return_type' (line 385)
        stypy_return_type_91069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_overflow'
        return stypy_return_type_91069


    @norecursion
    def test_regression_1310(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_regression_1310'
        module_type_store = module_type_store.open_function_context('test_regression_1310', 392, 4, False)
        # Assigning a type to the variable 'self' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_localization', localization)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_function_name', 'TestBisplrep.test_regression_1310')
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_param_names_list', [])
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestBisplrep.test_regression_1310.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBisplrep.test_regression_1310', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_regression_1310', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_regression_1310(...)' code ##################

        
        # Assigning a Subscript to a Name (line 394):
        
        # Assigning a Subscript to a Name (line 394):
        
        # Obtaining the type of the subscript
        str_91070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 50), 'str', 'data')
        
        # Call to load(...): (line 394)
        # Processing the call arguments (line 394)
        
        # Call to data_file(...): (line 394)
        # Processing the call arguments (line 394)
        str_91074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 33), 'str', 'bug-1310.npz')
        # Processing the call keyword arguments (line 394)
        kwargs_91075 = {}
        # Getting the type of 'data_file' (line 394)
        data_file_91073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'data_file', False)
        # Calling data_file(args, kwargs) (line 394)
        data_file_call_result_91076 = invoke(stypy.reporting.localization.Localization(__file__, 394, 23), data_file_91073, *[str_91074], **kwargs_91075)
        
        # Processing the call keyword arguments (line 394)
        kwargs_91077 = {}
        # Getting the type of 'np' (line 394)
        np_91071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 15), 'np', False)
        # Obtaining the member 'load' of a type (line 394)
        load_91072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), np_91071, 'load')
        # Calling load(args, kwargs) (line 394)
        load_call_result_91078 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), load_91072, *[data_file_call_result_91076], **kwargs_91077)
        
        # Obtaining the member '__getitem__' of a type (line 394)
        getitem___91079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 15), load_call_result_91078, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 394)
        subscript_call_result_91080 = invoke(stypy.reporting.localization.Localization(__file__, 394, 15), getitem___91079, str_91070)
        
        # Assigning a type to the variable 'data' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'data', subscript_call_result_91080)
        
        # Call to bisplrep(...): (line 400)
        # Processing the call arguments (line 400)
        
        # Obtaining the type of the subscript
        slice_91082 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 400, 17), None, None, None)
        int_91083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 24), 'int')
        # Getting the type of 'data' (line 400)
        data_91084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___91085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 17), data_91084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_91086 = invoke(stypy.reporting.localization.Localization(__file__, 400, 17), getitem___91085, (slice_91082, int_91083))
        
        
        # Obtaining the type of the subscript
        slice_91087 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 400, 28), None, None, None)
        int_91088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 35), 'int')
        # Getting the type of 'data' (line 400)
        data_91089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 28), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___91090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 28), data_91089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_91091 = invoke(stypy.reporting.localization.Localization(__file__, 400, 28), getitem___91090, (slice_91087, int_91088))
        
        
        # Obtaining the type of the subscript
        slice_91092 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 400, 39), None, None, None)
        int_91093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 46), 'int')
        # Getting the type of 'data' (line 400)
        data_91094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 400)
        getitem___91095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 39), data_91094, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 400)
        subscript_call_result_91096 = invoke(stypy.reporting.localization.Localization(__file__, 400, 39), getitem___91095, (slice_91092, int_91093))
        
        # Processing the call keyword arguments (line 400)
        int_91097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 53), 'int')
        keyword_91098 = int_91097
        int_91099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 59), 'int')
        keyword_91100 = int_91099
        int_91101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 64), 'int')
        keyword_91102 = int_91101
        # Getting the type of 'True' (line 401)
        True_91103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 29), 'True', False)
        keyword_91104 = True_91103
        kwargs_91105 = {'ky': keyword_91100, 'kx': keyword_91098, 's': keyword_91102, 'full_output': keyword_91104}
        # Getting the type of 'bisplrep' (line 400)
        bisplrep_91081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'bisplrep', False)
        # Calling bisplrep(args, kwargs) (line 400)
        bisplrep_call_result_91106 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), bisplrep_91081, *[subscript_call_result_91086, subscript_call_result_91091, subscript_call_result_91096], **kwargs_91105)
        
        
        # ################# End of 'test_regression_1310(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_regression_1310' in the type store
        # Getting the type of 'stypy_return_type' (line 392)
        stypy_return_type_91107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_regression_1310'
        return stypy_return_type_91107


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 384, 0, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestBisplrep.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestBisplrep' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 0), 'TestBisplrep', TestBisplrep)

@norecursion
def test_dblint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_dblint'
    module_type_store = module_type_store.open_function_context('test_dblint', 404, 0, False)
    
    # Passed parameters checking function
    test_dblint.stypy_localization = localization
    test_dblint.stypy_type_of_self = None
    test_dblint.stypy_type_store = module_type_store
    test_dblint.stypy_function_name = 'test_dblint'
    test_dblint.stypy_param_names_list = []
    test_dblint.stypy_varargs_param_name = None
    test_dblint.stypy_kwargs_param_name = None
    test_dblint.stypy_call_defaults = defaults
    test_dblint.stypy_call_varargs = varargs
    test_dblint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_dblint', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_dblint', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_dblint(...)' code ##################

    
    # Assigning a Call to a Name (line 407):
    
    # Assigning a Call to a Name (line 407):
    
    # Call to linspace(...): (line 407)
    # Processing the call arguments (line 407)
    int_91110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 20), 'int')
    int_91111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 23), 'int')
    # Processing the call keyword arguments (line 407)
    kwargs_91112 = {}
    # Getting the type of 'np' (line 407)
    np_91108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 407)
    linspace_91109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 8), np_91108, 'linspace')
    # Calling linspace(args, kwargs) (line 407)
    linspace_call_result_91113 = invoke(stypy.reporting.localization.Localization(__file__, 407, 8), linspace_91109, *[int_91110, int_91111], **kwargs_91112)
    
    # Assigning a type to the variable 'x' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'x', linspace_call_result_91113)
    
    # Assigning a Call to a Name (line 408):
    
    # Assigning a Call to a Name (line 408):
    
    # Call to linspace(...): (line 408)
    # Processing the call arguments (line 408)
    int_91116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 20), 'int')
    int_91117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 23), 'int')
    # Processing the call keyword arguments (line 408)
    kwargs_91118 = {}
    # Getting the type of 'np' (line 408)
    np_91114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 408)
    linspace_91115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), np_91114, 'linspace')
    # Calling linspace(args, kwargs) (line 408)
    linspace_call_result_91119 = invoke(stypy.reporting.localization.Localization(__file__, 408, 8), linspace_91115, *[int_91116, int_91117], **kwargs_91118)
    
    # Assigning a type to the variable 'y' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'y', linspace_call_result_91119)
    
    # Assigning a Call to a Tuple (line 409):
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_91120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'int')
    
    # Call to meshgrid(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'x' (line 409)
    x_91123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'x', False)
    # Getting the type of 'y' (line 409)
    y_91124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'y', False)
    # Processing the call keyword arguments (line 409)
    kwargs_91125 = {}
    # Getting the type of 'np' (line 409)
    np_91121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 409)
    meshgrid_91122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 13), np_91121, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 409)
    meshgrid_call_result_91126 = invoke(stypy.reporting.localization.Localization(__file__, 409, 13), meshgrid_91122, *[x_91123, y_91124], **kwargs_91125)
    
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___91127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 4), meshgrid_call_result_91126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_91128 = invoke(stypy.reporting.localization.Localization(__file__, 409, 4), getitem___91127, int_91120)
    
    # Assigning a type to the variable 'tuple_var_assignment_88961' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'tuple_var_assignment_88961', subscript_call_result_91128)
    
    # Assigning a Subscript to a Name (line 409):
    
    # Obtaining the type of the subscript
    int_91129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'int')
    
    # Call to meshgrid(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'x' (line 409)
    x_91132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 25), 'x', False)
    # Getting the type of 'y' (line 409)
    y_91133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'y', False)
    # Processing the call keyword arguments (line 409)
    kwargs_91134 = {}
    # Getting the type of 'np' (line 409)
    np_91130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 409)
    meshgrid_91131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 13), np_91130, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 409)
    meshgrid_call_result_91135 = invoke(stypy.reporting.localization.Localization(__file__, 409, 13), meshgrid_91131, *[x_91132, y_91133], **kwargs_91134)
    
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___91136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 4), meshgrid_call_result_91135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 409)
    subscript_call_result_91137 = invoke(stypy.reporting.localization.Localization(__file__, 409, 4), getitem___91136, int_91129)
    
    # Assigning a type to the variable 'tuple_var_assignment_88962' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'tuple_var_assignment_88962', subscript_call_result_91137)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_88961' (line 409)
    tuple_var_assignment_88961_91138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'tuple_var_assignment_88961')
    # Assigning a type to the variable 'xx' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'xx', tuple_var_assignment_88961_91138)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'tuple_var_assignment_88962' (line 409)
    tuple_var_assignment_88962_91139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'tuple_var_assignment_88962')
    # Assigning a type to the variable 'yy' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'yy', tuple_var_assignment_88962_91139)
    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to RectBivariateSpline(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'x' (line 410)
    x_91142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 43), 'x', False)
    # Getting the type of 'y' (line 410)
    y_91143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 46), 'y', False)
    int_91144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 49), 'int')
    # Getting the type of 'xx' (line 410)
    xx_91145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 53), 'xx', False)
    # Applying the binary operator '*' (line 410)
    result_mul_91146 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 49), '*', int_91144, xx_91145)
    
    # Getting the type of 'yy' (line 410)
    yy_91147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 58), 'yy', False)
    # Applying the binary operator '*' (line 410)
    result_mul_91148 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 56), '*', result_mul_91146, yy_91147)
    
    # Processing the call keyword arguments (line 410)
    kwargs_91149 = {}
    # Getting the type of 'interpolate' (line 410)
    interpolate_91140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 11), 'interpolate', False)
    # Obtaining the member 'RectBivariateSpline' of a type (line 410)
    RectBivariateSpline_91141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 11), interpolate_91140, 'RectBivariateSpline')
    # Calling RectBivariateSpline(args, kwargs) (line 410)
    RectBivariateSpline_call_result_91150 = invoke(stypy.reporting.localization.Localization(__file__, 410, 11), RectBivariateSpline_91141, *[x_91142, y_91143, result_mul_91148], **kwargs_91149)
    
    # Assigning a type to the variable 'rect' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'rect', RectBivariateSpline_call_result_91150)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to list(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'rect' (line 411)
    rect_91152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'rect', False)
    # Obtaining the member 'tck' of a type (line 411)
    tck_91153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 15), rect_91152, 'tck')
    # Processing the call keyword arguments (line 411)
    kwargs_91154 = {}
    # Getting the type of 'list' (line 411)
    list_91151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 10), 'list', False)
    # Calling list(args, kwargs) (line 411)
    list_call_result_91155 = invoke(stypy.reporting.localization.Localization(__file__, 411, 10), list_91151, *[tck_91153], **kwargs_91154)
    
    # Assigning a type to the variable 'tck' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'tck', list_call_result_91155)
    
    # Call to extend(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'rect' (line 412)
    rect_91158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 15), 'rect', False)
    # Obtaining the member 'degrees' of a type (line 412)
    degrees_91159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 15), rect_91158, 'degrees')
    # Processing the call keyword arguments (line 412)
    kwargs_91160 = {}
    # Getting the type of 'tck' (line 412)
    tck_91156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'tck', False)
    # Obtaining the member 'extend' of a type (line 412)
    extend_91157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 4), tck_91156, 'extend')
    # Calling extend(args, kwargs) (line 412)
    extend_call_result_91161 = invoke(stypy.reporting.localization.Localization(__file__, 412, 4), extend_91157, *[degrees_91159], **kwargs_91160)
    
    
    # Call to assert_almost_equal(...): (line 414)
    # Processing the call arguments (line 414)
    
    # Call to dblint(...): (line 414)
    # Processing the call arguments (line 414)
    int_91164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 31), 'int')
    int_91165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 34), 'int')
    int_91166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 37), 'int')
    int_91167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 40), 'int')
    # Getting the type of 'tck' (line 414)
    tck_91168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 43), 'tck', False)
    # Processing the call keyword arguments (line 414)
    kwargs_91169 = {}
    # Getting the type of 'dblint' (line 414)
    dblint_91163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 24), 'dblint', False)
    # Calling dblint(args, kwargs) (line 414)
    dblint_call_result_91170 = invoke(stypy.reporting.localization.Localization(__file__, 414, 24), dblint_91163, *[int_91164, int_91165, int_91166, int_91167, tck_91168], **kwargs_91169)
    
    int_91171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 49), 'int')
    # Processing the call keyword arguments (line 414)
    kwargs_91172 = {}
    # Getting the type of 'assert_almost_equal' (line 414)
    assert_almost_equal_91162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 414)
    assert_almost_equal_call_result_91173 = invoke(stypy.reporting.localization.Localization(__file__, 414, 4), assert_almost_equal_91162, *[dblint_call_result_91170, int_91171], **kwargs_91172)
    
    
    # Call to assert_almost_equal(...): (line 415)
    # Processing the call arguments (line 415)
    
    # Call to dblint(...): (line 415)
    # Processing the call arguments (line 415)
    int_91176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 31), 'int')
    float_91177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 34), 'float')
    int_91178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 39), 'int')
    int_91179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 42), 'int')
    # Getting the type of 'tck' (line 415)
    tck_91180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 45), 'tck', False)
    # Processing the call keyword arguments (line 415)
    kwargs_91181 = {}
    # Getting the type of 'dblint' (line 415)
    dblint_91175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 24), 'dblint', False)
    # Calling dblint(args, kwargs) (line 415)
    dblint_call_result_91182 = invoke(stypy.reporting.localization.Localization(__file__, 415, 24), dblint_91175, *[int_91176, float_91177, int_91178, int_91179, tck_91180], **kwargs_91181)
    
    float_91183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 51), 'float')
    # Processing the call keyword arguments (line 415)
    kwargs_91184 = {}
    # Getting the type of 'assert_almost_equal' (line 415)
    assert_almost_equal_91174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 415)
    assert_almost_equal_call_result_91185 = invoke(stypy.reporting.localization.Localization(__file__, 415, 4), assert_almost_equal_91174, *[dblint_call_result_91182, float_91183], **kwargs_91184)
    
    
    # Call to assert_almost_equal(...): (line 416)
    # Processing the call arguments (line 416)
    
    # Call to dblint(...): (line 416)
    # Processing the call arguments (line 416)
    float_91188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 31), 'float')
    int_91189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 36), 'int')
    int_91190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 39), 'int')
    int_91191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 42), 'int')
    # Getting the type of 'tck' (line 416)
    tck_91192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 45), 'tck', False)
    # Processing the call keyword arguments (line 416)
    kwargs_91193 = {}
    # Getting the type of 'dblint' (line 416)
    dblint_91187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 24), 'dblint', False)
    # Calling dblint(args, kwargs) (line 416)
    dblint_call_result_91194 = invoke(stypy.reporting.localization.Localization(__file__, 416, 24), dblint_91187, *[float_91188, int_91189, int_91190, int_91191, tck_91192], **kwargs_91193)
    
    float_91195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 51), 'float')
    # Processing the call keyword arguments (line 416)
    kwargs_91196 = {}
    # Getting the type of 'assert_almost_equal' (line 416)
    assert_almost_equal_91186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 416)
    assert_almost_equal_call_result_91197 = invoke(stypy.reporting.localization.Localization(__file__, 416, 4), assert_almost_equal_91186, *[dblint_call_result_91194, float_91195], **kwargs_91196)
    
    
    # Call to assert_almost_equal(...): (line 417)
    # Processing the call arguments (line 417)
    
    # Call to dblint(...): (line 417)
    # Processing the call arguments (line 417)
    int_91200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 31), 'int')
    int_91201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 37), 'int')
    int_91202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 42), 'int')
    int_91203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 48), 'int')
    # Getting the type of 'tck' (line 417)
    tck_91204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 53), 'tck', False)
    # Processing the call keyword arguments (line 417)
    kwargs_91205 = {}
    # Getting the type of 'dblint' (line 417)
    dblint_91199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 24), 'dblint', False)
    # Calling dblint(args, kwargs) (line 417)
    dblint_call_result_91206 = invoke(stypy.reporting.localization.Localization(__file__, 417, 24), dblint_91199, *[int_91200, int_91201, int_91202, int_91203, tck_91204], **kwargs_91205)
    
    int_91207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 59), 'int')
    # Processing the call keyword arguments (line 417)
    kwargs_91208 = {}
    # Getting the type of 'assert_almost_equal' (line 417)
    assert_almost_equal_91198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 417)
    assert_almost_equal_call_result_91209 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), assert_almost_equal_91198, *[dblint_call_result_91206, int_91207], **kwargs_91208)
    
    
    # ################# End of 'test_dblint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_dblint' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_91210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_dblint'
    return stypy_return_type_91210

# Assigning a type to the variable 'test_dblint' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'test_dblint', test_dblint)

@norecursion
def test_splev_der_k(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_splev_der_k'
    module_type_store = module_type_store.open_function_context('test_splev_der_k', 420, 0, False)
    
    # Passed parameters checking function
    test_splev_der_k.stypy_localization = localization
    test_splev_der_k.stypy_type_of_self = None
    test_splev_der_k.stypy_type_store = module_type_store
    test_splev_der_k.stypy_function_name = 'test_splev_der_k'
    test_splev_der_k.stypy_param_names_list = []
    test_splev_der_k.stypy_varargs_param_name = None
    test_splev_der_k.stypy_kwargs_param_name = None
    test_splev_der_k.stypy_call_defaults = defaults
    test_splev_der_k.stypy_call_varargs = varargs
    test_splev_der_k.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_splev_der_k', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_splev_der_k', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_splev_der_k(...)' code ##################

    
    # Assigning a Tuple to a Name (line 425):
    
    # Assigning a Tuple to a Name (line 425):
    
    # Obtaining an instance of the builtin type 'tuple' (line 425)
    tuple_91211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 425)
    # Adding element type (line 425)
    
    # Call to array(...): (line 425)
    # Processing the call arguments (line 425)
    
    # Obtaining an instance of the builtin type 'list' (line 425)
    list_91214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 425)
    # Adding element type (line 425)
    float_91215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 20), list_91214, float_91215)
    # Adding element type (line 425)
    float_91216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 20), list_91214, float_91216)
    # Adding element type (line 425)
    float_91217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 29), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 20), list_91214, float_91217)
    # Adding element type (line 425)
    float_91218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 20), list_91214, float_91218)
    
    # Processing the call keyword arguments (line 425)
    kwargs_91219 = {}
    # Getting the type of 'np' (line 425)
    np_91212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 425)
    array_91213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 11), np_91212, 'array')
    # Calling array(args, kwargs) (line 425)
    array_call_result_91220 = invoke(stypy.reporting.localization.Localization(__file__, 425, 11), array_91213, *[list_91214], **kwargs_91219)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 11), tuple_91211, array_call_result_91220)
    # Adding element type (line 425)
    
    # Call to array(...): (line 426)
    # Processing the call arguments (line 426)
    
    # Obtaining an instance of the builtin type 'list' (line 426)
    list_91223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 426)
    # Adding element type (line 426)
    float_91224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 20), list_91223, float_91224)
    # Adding element type (line 426)
    float_91225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 34), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 20), list_91223, float_91225)
    # Adding element type (line 426)
    float_91226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 46), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 20), list_91223, float_91226)
    # Adding element type (line 426)
    float_91227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 20), list_91223, float_91227)
    
    # Processing the call keyword arguments (line 426)
    kwargs_91228 = {}
    # Getting the type of 'np' (line 426)
    np_91221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 426)
    array_91222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 11), np_91221, 'array')
    # Calling array(args, kwargs) (line 426)
    array_call_result_91229 = invoke(stypy.reporting.localization.Localization(__file__, 426, 11), array_91222, *[list_91223], **kwargs_91228)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 11), tuple_91211, array_call_result_91229)
    # Adding element type (line 425)
    int_91230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 11), tuple_91211, int_91230)
    
    # Assigning a type to the variable 'tck' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'tck', tuple_91211)
    
    # Assigning a Name to a Tuple (line 428):
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    int_91231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 4), 'int')
    # Getting the type of 'tck' (line 428)
    tck_91232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___91233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 4), tck_91232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_91234 = invoke(stypy.reporting.localization.Localization(__file__, 428, 4), getitem___91233, int_91231)
    
    # Assigning a type to the variable 'tuple_var_assignment_88963' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88963', subscript_call_result_91234)
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    int_91235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 4), 'int')
    # Getting the type of 'tck' (line 428)
    tck_91236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___91237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 4), tck_91236, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_91238 = invoke(stypy.reporting.localization.Localization(__file__, 428, 4), getitem___91237, int_91235)
    
    # Assigning a type to the variable 'tuple_var_assignment_88964' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88964', subscript_call_result_91238)
    
    # Assigning a Subscript to a Name (line 428):
    
    # Obtaining the type of the subscript
    int_91239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 4), 'int')
    # Getting the type of 'tck' (line 428)
    tck_91240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 14), 'tck')
    # Obtaining the member '__getitem__' of a type (line 428)
    getitem___91241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 4), tck_91240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 428)
    subscript_call_result_91242 = invoke(stypy.reporting.localization.Localization(__file__, 428, 4), getitem___91241, int_91239)
    
    # Assigning a type to the variable 'tuple_var_assignment_88965' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88965', subscript_call_result_91242)
    
    # Assigning a Name to a Name (line 428):
    # Getting the type of 'tuple_var_assignment_88963' (line 428)
    tuple_var_assignment_88963_91243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88963')
    # Assigning a type to the variable 't' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 't', tuple_var_assignment_88963_91243)
    
    # Assigning a Name to a Name (line 428):
    # Getting the type of 'tuple_var_assignment_88964' (line 428)
    tuple_var_assignment_88964_91244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88964')
    # Assigning a type to the variable 'c' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 7), 'c', tuple_var_assignment_88964_91244)
    
    # Assigning a Name to a Name (line 428):
    # Getting the type of 'tuple_var_assignment_88965' (line 428)
    tuple_var_assignment_88965_91245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'tuple_var_assignment_88965')
    # Assigning a type to the variable 'k' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 10), 'k', tuple_var_assignment_88965_91245)
    
    # Assigning a Call to a Name (line 429):
    
    # Assigning a Call to a Name (line 429):
    
    # Call to array(...): (line 429)
    # Processing the call arguments (line 429)
    
    # Obtaining an instance of the builtin type 'list' (line 429)
    list_91248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 429)
    # Adding element type (line 429)
    int_91249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_91248, int_91249)
    # Adding element type (line 429)
    int_91250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_91248, int_91250)
    # Adding element type (line 429)
    float_91251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_91248, float_91251)
    # Adding element type (line 429)
    int_91252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_91248, int_91252)
    
    # Processing the call keyword arguments (line 429)
    kwargs_91253 = {}
    # Getting the type of 'np' (line 429)
    np_91246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 429)
    array_91247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), np_91246, 'array')
    # Calling array(args, kwargs) (line 429)
    array_call_result_91254 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), array_91247, *[list_91248], **kwargs_91253)
    
    # Assigning a type to the variable 'x' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'x', array_call_result_91254)
    
    # Call to assert_allclose(...): (line 432)
    # Processing the call arguments (line 432)
    
    # Call to splev(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'x' (line 432)
    x_91257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 26), 'x', False)
    # Getting the type of 'tck' (line 432)
    tck_91258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 29), 'tck', False)
    # Processing the call keyword arguments (line 432)
    kwargs_91259 = {}
    # Getting the type of 'splev' (line 432)
    splev_91256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 20), 'splev', False)
    # Calling splev(args, kwargs) (line 432)
    splev_call_result_91260 = invoke(stypy.reporting.localization.Localization(__file__, 432, 20), splev_91256, *[x_91257, tck_91258], **kwargs_91259)
    
    
    # Obtaining the type of the subscript
    int_91261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 37), 'int')
    # Getting the type of 'c' (line 432)
    c_91262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 35), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___91263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 35), c_91262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_91264 = invoke(stypy.reporting.localization.Localization(__file__, 432, 35), getitem___91263, int_91261)
    
    
    # Obtaining the type of the subscript
    int_91265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 45), 'int')
    # Getting the type of 'c' (line 432)
    c_91266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 43), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___91267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 43), c_91266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_91268 = invoke(stypy.reporting.localization.Localization(__file__, 432, 43), getitem___91267, int_91265)
    
    
    # Obtaining the type of the subscript
    int_91269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 52), 'int')
    # Getting the type of 'c' (line 432)
    c_91270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 50), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___91271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 50), c_91270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_91272 = invoke(stypy.reporting.localization.Localization(__file__, 432, 50), getitem___91271, int_91269)
    
    # Applying the binary operator '-' (line 432)
    result_sub_91273 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 43), '-', subscript_call_result_91268, subscript_call_result_91272)
    
    # Getting the type of 'x' (line 432)
    x_91274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 58), 'x', False)
    # Applying the binary operator '*' (line 432)
    result_mul_91275 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 42), '*', result_sub_91273, x_91274)
    
    
    # Obtaining the type of the subscript
    int_91276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 62), 'int')
    # Getting the type of 't' (line 432)
    t_91277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 60), 't', False)
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___91278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 60), t_91277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_91279 = invoke(stypy.reporting.localization.Localization(__file__, 432, 60), getitem___91278, int_91276)
    
    # Applying the binary operator 'div' (line 432)
    result_div_91280 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 59), 'div', result_mul_91275, subscript_call_result_91279)
    
    # Applying the binary operator '+' (line 432)
    result_add_91281 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 35), '+', subscript_call_result_91264, result_div_91280)
    
    # Processing the call keyword arguments (line 432)
    kwargs_91282 = {}
    # Getting the type of 'assert_allclose' (line 432)
    assert_allclose_91255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 432)
    assert_allclose_call_result_91283 = invoke(stypy.reporting.localization.Localization(__file__, 432, 4), assert_allclose_91255, *[splev_call_result_91260, result_add_91281], **kwargs_91282)
    
    
    # Call to assert_allclose(...): (line 433)
    # Processing the call arguments (line 433)
    
    # Call to splev(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'x' (line 433)
    x_91286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'x', False)
    # Getting the type of 'tck' (line 433)
    tck_91287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 29), 'tck', False)
    int_91288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 34), 'int')
    # Processing the call keyword arguments (line 433)
    kwargs_91289 = {}
    # Getting the type of 'splev' (line 433)
    splev_91285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 20), 'splev', False)
    # Calling splev(args, kwargs) (line 433)
    splev_call_result_91290 = invoke(stypy.reporting.localization.Localization(__file__, 433, 20), splev_91285, *[x_91286, tck_91287, int_91288], **kwargs_91289)
    
    
    # Obtaining the type of the subscript
    int_91291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 41), 'int')
    # Getting the type of 'c' (line 433)
    c_91292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 39), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 433)
    getitem___91293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 39), c_91292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 433)
    subscript_call_result_91294 = invoke(stypy.reporting.localization.Localization(__file__, 433, 39), getitem___91293, int_91291)
    
    
    # Obtaining the type of the subscript
    int_91295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 46), 'int')
    # Getting the type of 'c' (line 433)
    c_91296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 44), 'c', False)
    # Obtaining the member '__getitem__' of a type (line 433)
    getitem___91297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 44), c_91296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 433)
    subscript_call_result_91298 = invoke(stypy.reporting.localization.Localization(__file__, 433, 44), getitem___91297, int_91295)
    
    # Applying the binary operator '-' (line 433)
    result_sub_91299 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 39), '-', subscript_call_result_91294, subscript_call_result_91298)
    
    
    # Obtaining the type of the subscript
    int_91300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 54), 'int')
    # Getting the type of 't' (line 433)
    t_91301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 52), 't', False)
    # Obtaining the member '__getitem__' of a type (line 433)
    getitem___91302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 52), t_91301, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 433)
    subscript_call_result_91303 = invoke(stypy.reporting.localization.Localization(__file__, 433, 52), getitem___91302, int_91300)
    
    # Applying the binary operator 'div' (line 433)
    result_div_91304 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 38), 'div', result_sub_91299, subscript_call_result_91303)
    
    # Processing the call keyword arguments (line 433)
    kwargs_91305 = {}
    # Getting the type of 'assert_allclose' (line 433)
    assert_allclose_91284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 433)
    assert_allclose_call_result_91306 = invoke(stypy.reporting.localization.Localization(__file__, 433, 4), assert_allclose_91284, *[splev_call_result_91290, result_div_91304], **kwargs_91305)
    
    
    # Call to seed(...): (line 436)
    # Processing the call arguments (line 436)
    int_91310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 19), 'int')
    # Processing the call keyword arguments (line 436)
    kwargs_91311 = {}
    # Getting the type of 'np' (line 436)
    np_91307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 436)
    random_91308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), np_91307, 'random')
    # Obtaining the member 'seed' of a type (line 436)
    seed_91309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 4), random_91308, 'seed')
    # Calling seed(args, kwargs) (line 436)
    seed_call_result_91312 = invoke(stypy.reporting.localization.Localization(__file__, 436, 4), seed_91309, *[int_91310], **kwargs_91311)
    
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to sort(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Call to random(...): (line 437)
    # Processing the call arguments (line 437)
    int_91318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 33), 'int')
    # Processing the call keyword arguments (line 437)
    kwargs_91319 = {}
    # Getting the type of 'np' (line 437)
    np_91315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'np', False)
    # Obtaining the member 'random' of a type (line 437)
    random_91316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 16), np_91315, 'random')
    # Obtaining the member 'random' of a type (line 437)
    random_91317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 16), random_91316, 'random')
    # Calling random(args, kwargs) (line 437)
    random_call_result_91320 = invoke(stypy.reporting.localization.Localization(__file__, 437, 16), random_91317, *[int_91318], **kwargs_91319)
    
    # Processing the call keyword arguments (line 437)
    kwargs_91321 = {}
    # Getting the type of 'np' (line 437)
    np_91313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'np', False)
    # Obtaining the member 'sort' of a type (line 437)
    sort_91314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), np_91313, 'sort')
    # Calling sort(args, kwargs) (line 437)
    sort_call_result_91322 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), sort_91314, *[random_call_result_91320], **kwargs_91321)
    
    # Assigning a type to the variable 'x' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'x', sort_call_result_91322)
    
    # Assigning a Call to a Name (line 438):
    
    # Assigning a Call to a Name (line 438):
    
    # Call to random(...): (line 438)
    # Processing the call arguments (line 438)
    int_91326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 25), 'int')
    # Processing the call keyword arguments (line 438)
    kwargs_91327 = {}
    # Getting the type of 'np' (line 438)
    np_91323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 438)
    random_91324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), np_91323, 'random')
    # Obtaining the member 'random' of a type (line 438)
    random_91325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 8), random_91324, 'random')
    # Calling random(args, kwargs) (line 438)
    random_call_result_91328 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), random_91325, *[int_91326], **kwargs_91327)
    
    # Assigning a type to the variable 'y' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'y', random_call_result_91328)
    
    # Assigning a Call to a Tuple (line 439):
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_91329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'int')
    
    # Call to splrep(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'x' (line 439)
    x_91331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'x', False)
    # Getting the type of 'y' (line 439)
    y_91332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'y', False)
    # Processing the call keyword arguments (line 439)
    kwargs_91333 = {}
    # Getting the type of 'splrep' (line 439)
    splrep_91330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 14), 'splrep', False)
    # Calling splrep(args, kwargs) (line 439)
    splrep_call_result_91334 = invoke(stypy.reporting.localization.Localization(__file__, 439, 14), splrep_91330, *[x_91331, y_91332], **kwargs_91333)
    
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___91335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), splrep_call_result_91334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_91336 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___91335, int_91329)
    
    # Assigning a type to the variable 'tuple_var_assignment_88966' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88966', subscript_call_result_91336)
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_91337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'int')
    
    # Call to splrep(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'x' (line 439)
    x_91339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'x', False)
    # Getting the type of 'y' (line 439)
    y_91340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'y', False)
    # Processing the call keyword arguments (line 439)
    kwargs_91341 = {}
    # Getting the type of 'splrep' (line 439)
    splrep_91338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 14), 'splrep', False)
    # Calling splrep(args, kwargs) (line 439)
    splrep_call_result_91342 = invoke(stypy.reporting.localization.Localization(__file__, 439, 14), splrep_91338, *[x_91339, y_91340], **kwargs_91341)
    
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___91343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), splrep_call_result_91342, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_91344 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___91343, int_91337)
    
    # Assigning a type to the variable 'tuple_var_assignment_88967' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88967', subscript_call_result_91344)
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_91345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'int')
    
    # Call to splrep(...): (line 439)
    # Processing the call arguments (line 439)
    # Getting the type of 'x' (line 439)
    x_91347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'x', False)
    # Getting the type of 'y' (line 439)
    y_91348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 24), 'y', False)
    # Processing the call keyword arguments (line 439)
    kwargs_91349 = {}
    # Getting the type of 'splrep' (line 439)
    splrep_91346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 14), 'splrep', False)
    # Calling splrep(args, kwargs) (line 439)
    splrep_call_result_91350 = invoke(stypy.reporting.localization.Localization(__file__, 439, 14), splrep_91346, *[x_91347, y_91348], **kwargs_91349)
    
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___91351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), splrep_call_result_91350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_91352 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___91351, int_91345)
    
    # Assigning a type to the variable 'tuple_var_assignment_88968' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88968', subscript_call_result_91352)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_88966' (line 439)
    tuple_var_assignment_88966_91353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88966')
    # Assigning a type to the variable 't' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 't', tuple_var_assignment_88966_91353)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_88967' (line 439)
    tuple_var_assignment_88967_91354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88967')
    # Assigning a type to the variable 'c' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 7), 'c', tuple_var_assignment_88967_91354)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_88968' (line 439)
    tuple_var_assignment_88968_91355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_88968')
    # Assigning a type to the variable 'k' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 10), 'k', tuple_var_assignment_88968_91355)
    
    # Assigning a List to a Name (line 441):
    
    # Assigning a List to a Name (line 441):
    
    # Obtaining an instance of the builtin type 'list' (line 441)
    list_91356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 441)
    # Adding element type (line 441)
    
    # Obtaining the type of the subscript
    int_91357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 11), 'int')
    # Getting the type of 't' (line 441)
    t_91358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 9), 't')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___91359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 9), t_91358, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_91360 = invoke(stypy.reporting.localization.Localization(__file__, 441, 9), getitem___91359, int_91357)
    
    float_91361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 16), 'float')
    # Applying the binary operator '-' (line 441)
    result_sub_91362 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 9), '-', subscript_call_result_91360, float_91361)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 8), list_91356, result_sub_91362)
    # Adding element type (line 441)
    
    # Obtaining the type of the subscript
    int_91363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 22), 'int')
    # Getting the type of 't' (line 441)
    t_91364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 't')
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___91365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 20), t_91364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_91366 = invoke(stypy.reporting.localization.Localization(__file__, 441, 20), getitem___91365, int_91363)
    
    float_91367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 28), 'float')
    # Applying the binary operator '+' (line 441)
    result_add_91368 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 20), '+', subscript_call_result_91366, float_91367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 8), list_91356, result_add_91368)
    
    # Assigning a type to the variable 'x' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'x', list_91356)
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to splder(...): (line 442)
    # Processing the call arguments (line 442)
    
    # Obtaining an instance of the builtin type 'tuple' (line 442)
    tuple_91370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 442)
    # Adding element type (line 442)
    # Getting the type of 't' (line 442)
    t_91371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), tuple_91370, t_91371)
    # Adding element type (line 442)
    # Getting the type of 'c' (line 442)
    c_91372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 22), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), tuple_91370, c_91372)
    # Adding element type (line 442)
    # Getting the type of 'k' (line 442)
    k_91373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 25), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 442, 19), tuple_91370, k_91373)
    
    # Getting the type of 'k' (line 442)
    k_91374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 29), 'k', False)
    # Processing the call keyword arguments (line 442)
    kwargs_91375 = {}
    # Getting the type of 'splder' (line 442)
    splder_91369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 11), 'splder', False)
    # Calling splder(args, kwargs) (line 442)
    splder_call_result_91376 = invoke(stypy.reporting.localization.Localization(__file__, 442, 11), splder_91369, *[tuple_91370, k_91374], **kwargs_91375)
    
    # Assigning a type to the variable 'tck2' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tck2', splder_call_result_91376)
    
    # Call to assert_allclose(...): (line 443)
    # Processing the call arguments (line 443)
    
    # Call to splev(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'x' (line 443)
    x_91379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 26), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 443)
    tuple_91380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 443)
    # Adding element type (line 443)
    # Getting the type of 't' (line 443)
    t_91381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 30), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 30), tuple_91380, t_91381)
    # Adding element type (line 443)
    # Getting the type of 'c' (line 443)
    c_91382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 33), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 30), tuple_91380, c_91382)
    # Adding element type (line 443)
    # Getting the type of 'k' (line 443)
    k_91383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 36), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 30), tuple_91380, k_91383)
    
    # Getting the type of 'k' (line 443)
    k_91384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 40), 'k', False)
    # Processing the call keyword arguments (line 443)
    kwargs_91385 = {}
    # Getting the type of 'splev' (line 443)
    splev_91378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'splev', False)
    # Calling splev(args, kwargs) (line 443)
    splev_call_result_91386 = invoke(stypy.reporting.localization.Localization(__file__, 443, 20), splev_91378, *[x_91379, tuple_91380, k_91384], **kwargs_91385)
    
    
    # Call to splev(...): (line 443)
    # Processing the call arguments (line 443)
    # Getting the type of 'x' (line 443)
    x_91388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 50), 'x', False)
    # Getting the type of 'tck2' (line 443)
    tck2_91389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 53), 'tck2', False)
    # Processing the call keyword arguments (line 443)
    kwargs_91390 = {}
    # Getting the type of 'splev' (line 443)
    splev_91387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 44), 'splev', False)
    # Calling splev(args, kwargs) (line 443)
    splev_call_result_91391 = invoke(stypy.reporting.localization.Localization(__file__, 443, 44), splev_91387, *[x_91388, tck2_91389], **kwargs_91390)
    
    # Processing the call keyword arguments (line 443)
    kwargs_91392 = {}
    # Getting the type of 'assert_allclose' (line 443)
    assert_allclose_91377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 443)
    assert_allclose_call_result_91393 = invoke(stypy.reporting.localization.Localization(__file__, 443, 4), assert_allclose_91377, *[splev_call_result_91386, splev_call_result_91391], **kwargs_91392)
    
    
    # ################# End of 'test_splev_der_k(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_splev_der_k' in the type store
    # Getting the type of 'stypy_return_type' (line 420)
    stypy_return_type_91394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_splev_der_k'
    return stypy_return_type_91394

# Assigning a type to the variable 'test_splev_der_k' (line 420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'test_splev_der_k', test_splev_der_k)

@norecursion
def test_bisplev_integer_overflow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_bisplev_integer_overflow'
    module_type_store = module_type_store.open_function_context('test_bisplev_integer_overflow', 446, 0, False)
    
    # Passed parameters checking function
    test_bisplev_integer_overflow.stypy_localization = localization
    test_bisplev_integer_overflow.stypy_type_of_self = None
    test_bisplev_integer_overflow.stypy_type_store = module_type_store
    test_bisplev_integer_overflow.stypy_function_name = 'test_bisplev_integer_overflow'
    test_bisplev_integer_overflow.stypy_param_names_list = []
    test_bisplev_integer_overflow.stypy_varargs_param_name = None
    test_bisplev_integer_overflow.stypy_kwargs_param_name = None
    test_bisplev_integer_overflow.stypy_call_defaults = defaults
    test_bisplev_integer_overflow.stypy_call_varargs = varargs
    test_bisplev_integer_overflow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_bisplev_integer_overflow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_bisplev_integer_overflow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_bisplev_integer_overflow(...)' code ##################

    
    # Call to seed(...): (line 447)
    # Processing the call arguments (line 447)
    int_91398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 19), 'int')
    # Processing the call keyword arguments (line 447)
    kwargs_91399 = {}
    # Getting the type of 'np' (line 447)
    np_91395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 447)
    random_91396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), np_91395, 'random')
    # Obtaining the member 'seed' of a type (line 447)
    seed_91397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 4), random_91396, 'seed')
    # Calling seed(args, kwargs) (line 447)
    seed_call_result_91400 = invoke(stypy.reporting.localization.Localization(__file__, 447, 4), seed_91397, *[int_91398], **kwargs_91399)
    
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to linspace(...): (line 449)
    # Processing the call arguments (line 449)
    int_91403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 20), 'int')
    int_91404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 23), 'int')
    int_91405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 26), 'int')
    # Processing the call keyword arguments (line 449)
    kwargs_91406 = {}
    # Getting the type of 'np' (line 449)
    np_91401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'np', False)
    # Obtaining the member 'linspace' of a type (line 449)
    linspace_91402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), np_91401, 'linspace')
    # Calling linspace(args, kwargs) (line 449)
    linspace_call_result_91407 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), linspace_91402, *[int_91403, int_91404, int_91405], **kwargs_91406)
    
    # Assigning a type to the variable 'x' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'x', linspace_call_result_91407)
    
    # Assigning a Name to a Name (line 450):
    
    # Assigning a Name to a Name (line 450):
    # Getting the type of 'x' (line 450)
    x_91408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'x')
    # Assigning a type to the variable 'y' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'y', x_91408)
    
    # Assigning a Call to a Name (line 451):
    
    # Assigning a Call to a Name (line 451):
    
    # Call to ravel(...): (line 451)
    # Processing the call keyword arguments (line 451)
    kwargs_91417 = {}
    
    # Call to randn(...): (line 451)
    # Processing the call arguments (line 451)
    int_91412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 24), 'int')
    int_91413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 28), 'int')
    # Processing the call keyword arguments (line 451)
    kwargs_91414 = {}
    # Getting the type of 'np' (line 451)
    np_91409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 451)
    random_91410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), np_91409, 'random')
    # Obtaining the member 'randn' of a type (line 451)
    randn_91411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), random_91410, 'randn')
    # Calling randn(args, kwargs) (line 451)
    randn_call_result_91415 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), randn_91411, *[int_91412, int_91413], **kwargs_91414)
    
    # Obtaining the member 'ravel' of a type (line 451)
    ravel_91416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), randn_call_result_91415, 'ravel')
    # Calling ravel(args, kwargs) (line 451)
    ravel_call_result_91418 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), ravel_91416, *[], **kwargs_91417)
    
    # Assigning a type to the variable 'z' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'z', ravel_call_result_91418)
    
    # Assigning a Num to a Name (line 452):
    
    # Assigning a Num to a Name (line 452):
    int_91419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 9), 'int')
    # Assigning a type to the variable 'kx' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'kx', int_91419)
    
    # Assigning a Num to a Name (line 453):
    
    # Assigning a Num to a Name (line 453):
    int_91420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 9), 'int')
    # Assigning a type to the variable 'ky' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'ky', int_91420)
    
    # Assigning a Call to a Tuple (line 455):
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91431 = kx_91430
    # Getting the type of 'ky' (line 456)
    ky_91432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91433 = ky_91432
    float_91434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91435 = float_91434
    kwargs_91436 = {'ky': keyword_91433, 'kx': keyword_91431, 's': keyword_91435}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91437 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91422, *[x_91423, y_91424, z_91425, None_91426, None_91427, None_91428, None_91429], **kwargs_91436)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91439 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91438, int_91421)
    
    # Assigning a type to the variable 'tuple_var_assignment_88969' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88969', subscript_call_result_91439)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91450 = kx_91449
    # Getting the type of 'ky' (line 456)
    ky_91451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91452 = ky_91451
    float_91453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91454 = float_91453
    kwargs_91455 = {'ky': keyword_91452, 'kx': keyword_91450, 's': keyword_91454}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91456 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91441, *[x_91442, y_91443, z_91444, None_91445, None_91446, None_91447, None_91448], **kwargs_91455)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91458 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91457, int_91440)
    
    # Assigning a type to the variable 'tuple_var_assignment_88970' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88970', subscript_call_result_91458)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91469 = kx_91468
    # Getting the type of 'ky' (line 456)
    ky_91470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91471 = ky_91470
    float_91472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91473 = float_91472
    kwargs_91474 = {'ky': keyword_91471, 'kx': keyword_91469, 's': keyword_91473}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91475 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91460, *[x_91461, y_91462, z_91463, None_91464, None_91465, None_91466, None_91467], **kwargs_91474)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91477 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91476, int_91459)
    
    # Assigning a type to the variable 'tuple_var_assignment_88971' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88971', subscript_call_result_91477)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91488 = kx_91487
    # Getting the type of 'ky' (line 456)
    ky_91489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91490 = ky_91489
    float_91491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91492 = float_91491
    kwargs_91493 = {'ky': keyword_91490, 'kx': keyword_91488, 's': keyword_91492}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91494 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91479, *[x_91480, y_91481, z_91482, None_91483, None_91484, None_91485, None_91486], **kwargs_91493)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91496 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91495, int_91478)
    
    # Assigning a type to the variable 'tuple_var_assignment_88972' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88972', subscript_call_result_91496)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91507 = kx_91506
    # Getting the type of 'ky' (line 456)
    ky_91508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91509 = ky_91508
    float_91510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91511 = float_91510
    kwargs_91512 = {'ky': keyword_91509, 'kx': keyword_91507, 's': keyword_91511}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91513 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91498, *[x_91499, y_91500, z_91501, None_91502, None_91503, None_91504, None_91505], **kwargs_91512)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91515 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91514, int_91497)
    
    # Assigning a type to the variable 'tuple_var_assignment_88973' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88973', subscript_call_result_91515)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91526 = kx_91525
    # Getting the type of 'ky' (line 456)
    ky_91527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91528 = ky_91527
    float_91529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91530 = float_91529
    kwargs_91531 = {'ky': keyword_91528, 'kx': keyword_91526, 's': keyword_91530}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91532 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91517, *[x_91518, y_91519, z_91520, None_91521, None_91522, None_91523, None_91524], **kwargs_91531)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91534 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91533, int_91516)
    
    # Assigning a type to the variable 'tuple_var_assignment_88974' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88974', subscript_call_result_91534)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_91535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 4), 'int')
    
    # Call to regrid_smth(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'x' (line 456)
    x_91537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'x', False)
    # Getting the type of 'y' (line 456)
    y_91538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'y', False)
    # Getting the type of 'z' (line 456)
    z_91539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 14), 'z', False)
    # Getting the type of 'None' (line 456)
    None_91540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 29), 'None', False)
    # Getting the type of 'None' (line 456)
    None_91543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'None', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'kx' (line 456)
    kx_91544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 44), 'kx', False)
    keyword_91545 = kx_91544
    # Getting the type of 'ky' (line 456)
    ky_91546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 51), 'ky', False)
    keyword_91547 = ky_91546
    float_91548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 57), 'float')
    keyword_91549 = float_91548
    kwargs_91550 = {'ky': keyword_91547, 'kx': keyword_91545, 's': keyword_91549}
    # Getting the type of 'regrid_smth' (line 455)
    regrid_smth_91536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 33), 'regrid_smth', False)
    # Calling regrid_smth(args, kwargs) (line 455)
    regrid_smth_call_result_91551 = invoke(stypy.reporting.localization.Localization(__file__, 455, 33), regrid_smth_91536, *[x_91537, y_91538, z_91539, None_91540, None_91541, None_91542, None_91543], **kwargs_91550)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___91552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 4), regrid_smth_call_result_91551, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_91553 = invoke(stypy.reporting.localization.Localization(__file__, 455, 4), getitem___91552, int_91535)
    
    # Assigning a type to the variable 'tuple_var_assignment_88975' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88975', subscript_call_result_91553)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88969' (line 455)
    tuple_var_assignment_88969_91554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88969')
    # Assigning a type to the variable 'nx' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'nx', tuple_var_assignment_88969_91554)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88970' (line 455)
    tuple_var_assignment_88970_91555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88970')
    # Assigning a type to the variable 'tx' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tx', tuple_var_assignment_88970_91555)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88971' (line 455)
    tuple_var_assignment_88971_91556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88971')
    # Assigning a type to the variable 'ny' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'ny', tuple_var_assignment_88971_91556)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88972' (line 455)
    tuple_var_assignment_88972_91557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88972')
    # Assigning a type to the variable 'ty' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 16), 'ty', tuple_var_assignment_88972_91557)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88973' (line 455)
    tuple_var_assignment_88973_91558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88973')
    # Assigning a type to the variable 'c' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'c', tuple_var_assignment_88973_91558)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88974' (line 455)
    tuple_var_assignment_88974_91559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88974')
    # Assigning a type to the variable 'fp' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 23), 'fp', tuple_var_assignment_88974_91559)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_88975' (line 455)
    tuple_var_assignment_88975_91560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'tuple_var_assignment_88975')
    # Assigning a type to the variable 'ier' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 27), 'ier', tuple_var_assignment_88975_91560)
    
    # Assigning a Tuple to a Name (line 457):
    
    # Assigning a Tuple to a Name (line 457):
    
    # Obtaining an instance of the builtin type 'tuple' (line 457)
    tuple_91561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 457)
    # Adding element type (line 457)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nx' (line 457)
    nx_91562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'nx')
    slice_91563 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 11), None, nx_91562, None)
    # Getting the type of 'tx' (line 457)
    tx_91564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 11), 'tx')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___91565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 11), tx_91564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_91566 = invoke(stypy.reporting.localization.Localization(__file__, 457, 11), getitem___91565, slice_91563)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 11), tuple_91561, subscript_call_result_91566)
    # Adding element type (line 457)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ny' (line 457)
    ny_91567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'ny')
    slice_91568 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 20), None, ny_91567, None)
    # Getting the type of 'ty' (line 457)
    ty_91569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 20), 'ty')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___91570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 20), ty_91569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_91571 = invoke(stypy.reporting.localization.Localization(__file__, 457, 20), getitem___91570, slice_91568)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 11), tuple_91561, subscript_call_result_91571)
    # Adding element type (line 457)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nx' (line 457)
    nx_91572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 33), 'nx')
    # Getting the type of 'kx' (line 457)
    kx_91573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 38), 'kx')
    # Applying the binary operator '-' (line 457)
    result_sub_91574 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 33), '-', nx_91572, kx_91573)
    
    int_91575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 43), 'int')
    # Applying the binary operator '-' (line 457)
    result_sub_91576 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 41), '-', result_sub_91574, int_91575)
    
    # Getting the type of 'ny' (line 457)
    ny_91577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'ny')
    # Getting the type of 'ky' (line 457)
    ky_91578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 54), 'ky')
    # Applying the binary operator '-' (line 457)
    result_sub_91579 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 49), '-', ny_91577, ky_91578)
    
    int_91580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 59), 'int')
    # Applying the binary operator '-' (line 457)
    result_sub_91581 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 57), '-', result_sub_91579, int_91580)
    
    # Applying the binary operator '*' (line 457)
    result_mul_91582 = python_operator(stypy.reporting.localization.Localization(__file__, 457, 32), '*', result_sub_91576, result_sub_91581)
    
    slice_91583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 457, 29), None, result_mul_91582, None)
    # Getting the type of 'c' (line 457)
    c_91584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 29), 'c')
    # Obtaining the member '__getitem__' of a type (line 457)
    getitem___91585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 29), c_91584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 457)
    subscript_call_result_91586 = invoke(stypy.reporting.localization.Localization(__file__, 457, 29), getitem___91585, slice_91583)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 11), tuple_91561, subscript_call_result_91586)
    # Adding element type (line 457)
    # Getting the type of 'kx' (line 457)
    kx_91587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 64), 'kx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 11), tuple_91561, kx_91587)
    # Adding element type (line 457)
    # Getting the type of 'ky' (line 457)
    ky_91588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 68), 'ky')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 11), tuple_91561, ky_91588)
    
    # Assigning a type to the variable 'tck' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'tck', tuple_91561)
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to zeros(...): (line 459)
    # Processing the call arguments (line 459)
    
    # Obtaining an instance of the builtin type 'list' (line 459)
    list_91591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 459)
    # Adding element type (line 459)
    int_91592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 18), list_91591, int_91592)
    
    # Processing the call keyword arguments (line 459)
    kwargs_91593 = {}
    # Getting the type of 'np' (line 459)
    np_91589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 459)
    zeros_91590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 9), np_91589, 'zeros')
    # Calling zeros(args, kwargs) (line 459)
    zeros_call_result_91594 = invoke(stypy.reporting.localization.Localization(__file__, 459, 9), zeros_91590, *[list_91591], **kwargs_91593)
    
    # Assigning a type to the variable 'xp' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'xp', zeros_call_result_91594)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to zeros(...): (line 460)
    # Processing the call arguments (line 460)
    
    # Obtaining an instance of the builtin type 'list' (line 460)
    list_91597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 460)
    # Adding element type (line 460)
    int_91598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 18), list_91597, int_91598)
    
    # Processing the call keyword arguments (line 460)
    kwargs_91599 = {}
    # Getting the type of 'np' (line 460)
    np_91595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 9), 'np', False)
    # Obtaining the member 'zeros' of a type (line 460)
    zeros_91596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 9), np_91595, 'zeros')
    # Calling zeros(args, kwargs) (line 460)
    zeros_call_result_91600 = invoke(stypy.reporting.localization.Localization(__file__, 460, 9), zeros_91596, *[list_91597], **kwargs_91599)
    
    # Assigning a type to the variable 'yp' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'yp', zeros_call_result_91600)
    
    # Call to assert_raises(...): (line 462)
    # Processing the call arguments (line 462)
    
    # Obtaining an instance of the builtin type 'tuple' (line 462)
    tuple_91602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 462)
    # Adding element type (line 462)
    # Getting the type of 'RuntimeError' (line 462)
    RuntimeError_91603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 19), 'RuntimeError', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), tuple_91602, RuntimeError_91603)
    # Adding element type (line 462)
    # Getting the type of 'MemoryError' (line 462)
    MemoryError_91604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 33), 'MemoryError', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 19), tuple_91602, MemoryError_91604)
    
    # Getting the type of 'bisplev' (line 462)
    bisplev_91605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 47), 'bisplev', False)
    # Getting the type of 'xp' (line 462)
    xp_91606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'xp', False)
    # Getting the type of 'yp' (line 462)
    yp_91607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 60), 'yp', False)
    # Getting the type of 'tck' (line 462)
    tck_91608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 64), 'tck', False)
    # Processing the call keyword arguments (line 462)
    kwargs_91609 = {}
    # Getting the type of 'assert_raises' (line 462)
    assert_raises_91601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 462)
    assert_raises_call_result_91610 = invoke(stypy.reporting.localization.Localization(__file__, 462, 4), assert_raises_91601, *[tuple_91602, bisplev_91605, xp_91606, yp_91607, tck_91608], **kwargs_91609)
    
    
    # ################# End of 'test_bisplev_integer_overflow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_bisplev_integer_overflow' in the type store
    # Getting the type of 'stypy_return_type' (line 446)
    stypy_return_type_91611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_91611)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_bisplev_integer_overflow'
    return stypy_return_type_91611

# Assigning a type to the variable 'test_bisplev_integer_overflow' (line 446)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'test_bisplev_integer_overflow', test_bisplev_integer_overflow)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
