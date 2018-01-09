
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: import numpy as np
6: from numpy.testing import assert_equal, assert_allclose, assert_almost_equal
7: from pytest import raises as assert_raises
8: from scipy._lib._numpy_compat import suppress_warnings
9: 
10: import scipy.interpolate.interpnd as interpnd
11: import scipy.spatial.qhull as qhull
12: 
13: import pickle
14: 
15: 
16: def data_file(basename):
17:     return os.path.join(os.path.abspath(os.path.dirname(__file__)),
18:                         'data', basename)
19: 
20: 
21: class TestLinearNDInterpolation(object):
22:     def test_smoketest(self):
23:         # Test at single points
24:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
25:                      dtype=np.double)
26:         y = np.arange(x.shape[0], dtype=np.double)
27: 
28:         yi = interpnd.LinearNDInterpolator(x, y)(x)
29:         assert_almost_equal(y, yi)
30: 
31:     def test_smoketest_alternate(self):
32:         # Test at single points, alternate calling convention
33:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
34:                      dtype=np.double)
35:         y = np.arange(x.shape[0], dtype=np.double)
36: 
37:         yi = interpnd.LinearNDInterpolator((x[:,0], x[:,1]), y)(x[:,0], x[:,1])
38:         assert_almost_equal(y, yi)
39: 
40:     def test_complex_smoketest(self):
41:         # Test at single points
42:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
43:                      dtype=np.double)
44:         y = np.arange(x.shape[0], dtype=np.double)
45:         y = y - 3j*y
46: 
47:         yi = interpnd.LinearNDInterpolator(x, y)(x)
48:         assert_almost_equal(y, yi)
49: 
50:     def test_tri_input(self):
51:         # Test at single points
52:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
53:                      dtype=np.double)
54:         y = np.arange(x.shape[0], dtype=np.double)
55:         y = y - 3j*y
56: 
57:         tri = qhull.Delaunay(x)
58:         yi = interpnd.LinearNDInterpolator(tri, y)(x)
59:         assert_almost_equal(y, yi)
60: 
61:     def test_square(self):
62:         # Test barycentric interpolation on a square against a manual
63:         # implementation
64: 
65:         points = np.array([(0,0), (0,1), (1,1), (1,0)], dtype=np.double)
66:         values = np.array([1., 2., -3., 5.], dtype=np.double)
67: 
68:         # NB: assume triangles (0, 1, 3) and (1, 2, 3)
69:         #
70:         #  1----2
71:         #  | \  |
72:         #  |  \ |
73:         #  0----3
74: 
75:         def ip(x, y):
76:             t1 = (x + y <= 1)
77:             t2 = ~t1
78: 
79:             x1 = x[t1]
80:             y1 = y[t1]
81: 
82:             x2 = x[t2]
83:             y2 = y[t2]
84: 
85:             z = 0*x
86: 
87:             z[t1] = (values[0]*(1 - x1 - y1)
88:                      + values[1]*y1
89:                      + values[3]*x1)
90: 
91:             z[t2] = (values[2]*(x2 + y2 - 1)
92:                      + values[1]*(1 - x2)
93:                      + values[3]*(1 - y2))
94:             return z
95: 
96:         xx, yy = np.broadcast_arrays(np.linspace(0, 1, 14)[:,None],
97:                                      np.linspace(0, 1, 14)[None,:])
98:         xx = xx.ravel()
99:         yy = yy.ravel()
100: 
101:         xi = np.array([xx, yy]).T.copy()
102:         zi = interpnd.LinearNDInterpolator(points, values)(xi)
103: 
104:         assert_almost_equal(zi, ip(xx, yy))
105: 
106:     def test_smoketest_rescale(self):
107:         # Test at single points
108:         x = np.array([(0, 0), (-5, -5), (-5, 5), (5, 5), (2.5, 3)],
109:                      dtype=np.double)
110:         y = np.arange(x.shape[0], dtype=np.double)
111: 
112:         yi = interpnd.LinearNDInterpolator(x, y, rescale=True)(x)
113:         assert_almost_equal(y, yi)
114: 
115:     def test_square_rescale(self):
116:         # Test barycentric interpolation on a rectangle with rescaling
117:         # agaings the same implementation without rescaling
118: 
119:         points = np.array([(0,0), (0,100), (10,100), (10,0)], dtype=np.double)
120:         values = np.array([1., 2., -3., 5.], dtype=np.double)
121: 
122:         xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
123:                                      np.linspace(0, 100, 14)[None,:])
124:         xx = xx.ravel()
125:         yy = yy.ravel()
126:         xi = np.array([xx, yy]).T.copy()
127:         zi = interpnd.LinearNDInterpolator(points, values)(xi)
128:         zi_rescaled = interpnd.LinearNDInterpolator(points, values,
129:                 rescale=True)(xi)
130: 
131:         assert_almost_equal(zi, zi_rescaled)
132: 
133:     def test_tripoints_input_rescale(self):
134:         # Test at single points
135:         x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
136:                      dtype=np.double)
137:         y = np.arange(x.shape[0], dtype=np.double)
138:         y = y - 3j*y
139: 
140:         tri = qhull.Delaunay(x)
141:         yi = interpnd.LinearNDInterpolator(tri.points, y)(x)
142:         yi_rescale = interpnd.LinearNDInterpolator(tri.points, y,
143:                 rescale=True)(x)
144:         assert_almost_equal(yi, yi_rescale)
145: 
146:     def test_tri_input_rescale(self):
147:         # Test at single points
148:         x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
149:                      dtype=np.double)
150:         y = np.arange(x.shape[0], dtype=np.double)
151:         y = y - 3j*y
152: 
153:         tri = qhull.Delaunay(x)
154:         try:
155:             interpnd.LinearNDInterpolator(tri, y, rescale=True)(x)
156:         except ValueError as e:
157:             if str(e) != ("Rescaling is not supported when passing a "
158:                           "Delaunay triangulation as ``points``."):
159:                 raise
160:         except:
161:             raise
162: 
163:     def test_pickle(self):
164:         # Test at single points
165:         np.random.seed(1234)
166:         x = np.random.rand(30, 2)
167:         y = np.random.rand(30) + 1j*np.random.rand(30)
168: 
169:         ip = interpnd.LinearNDInterpolator(x, y)
170:         ip2 = pickle.loads(pickle.dumps(ip))
171: 
172:         assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))
173: 
174: 
175: class TestEstimateGradients2DGlobal(object):
176:     def test_smoketest(self):
177:         x = np.array([(0, 0), (0, 2),
178:                       (1, 0), (1, 2), (0.25, 0.75), (0.6, 0.8)], dtype=float)
179:         tri = qhull.Delaunay(x)
180: 
181:         # Should be exact for linear functions, independent of triangulation
182: 
183:         funcs = [
184:             (lambda x, y: 0*x + 1, (0, 0)),
185:             (lambda x, y: 0 + x, (1, 0)),
186:             (lambda x, y: -2 + y, (0, 1)),
187:             (lambda x, y: 3 + 3*x + 14.15*y, (3, 14.15))
188:         ]
189: 
190:         for j, (func, grad) in enumerate(funcs):
191:             z = func(x[:,0], x[:,1])
192:             dz = interpnd.estimate_gradients_2d_global(tri, z, tol=1e-6)
193: 
194:             assert_equal(dz.shape, (6, 2))
195:             assert_allclose(dz, np.array(grad)[None,:] + 0*dz,
196:                             rtol=1e-5, atol=1e-5, err_msg="item %d" % j)
197: 
198:     def test_regression_2359(self):
199:         # Check regression --- for certain point sets, gradient
200:         # estimation could end up in an infinite loop
201:         points = np.load(data_file('estimate_gradients_hang.npy'))
202:         values = np.random.rand(points.shape[0])
203:         tri = qhull.Delaunay(points)
204: 
205:         # This should not hang
206:         with suppress_warnings() as sup:
207:             sup.filter(interpnd.GradientEstimationWarning,
208:                        "Gradient estimation did not converge")
209:             interpnd.estimate_gradients_2d_global(tri, values, maxiter=1)
210: 
211: 
212: class TestCloughTocher2DInterpolator(object):
213: 
214:     def _check_accuracy(self, func, x=None, tol=1e-6, alternate=False, rescale=False, **kw):
215:         np.random.seed(1234)
216:         if x is None:
217:             x = np.array([(0, 0), (0, 1),
218:                           (1, 0), (1, 1), (0.25, 0.75), (0.6, 0.8),
219:                           (0.5, 0.2)],
220:                          dtype=float)
221: 
222:         if not alternate:
223:             ip = interpnd.CloughTocher2DInterpolator(x, func(x[:,0], x[:,1]),
224:                                                      tol=1e-6, rescale=rescale)
225:         else:
226:             ip = interpnd.CloughTocher2DInterpolator((x[:,0], x[:,1]),
227:                                                      func(x[:,0], x[:,1]),
228:                                                      tol=1e-6, rescale=rescale)
229: 
230:         p = np.random.rand(50, 2)
231: 
232:         if not alternate:
233:             a = ip(p)
234:         else:
235:             a = ip(p[:,0], p[:,1])
236:         b = func(p[:,0], p[:,1])
237: 
238:         try:
239:             assert_allclose(a, b, **kw)
240:         except AssertionError:
241:             print(abs(a - b))
242:             print(ip.grad)
243:             raise
244: 
245:     def test_linear_smoketest(self):
246:         # Should be exact for linear functions, independent of triangulation
247:         funcs = [
248:             lambda x, y: 0*x + 1,
249:             lambda x, y: 0 + x,
250:             lambda x, y: -2 + y,
251:             lambda x, y: 3 + 3*x + 14.15*y,
252:         ]
253: 
254:         for j, func in enumerate(funcs):
255:             self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
256:                                  err_msg="Function %d" % j)
257:             self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
258:                                  alternate=True,
259:                                  err_msg="Function (alternate) %d" % j)
260:             # check rescaling
261:             self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
262:                                  err_msg="Function (rescaled) %d" % j, rescale=True)
263:             self._check_accuracy(func, tol=1e-13, atol=1e-7, rtol=1e-7,
264:                                  alternate=True, rescale=True,
265:                                  err_msg="Function (alternate, rescaled) %d" % j)
266: 
267:     def test_quadratic_smoketest(self):
268:         # Should be reasonably accurate for quadratic functions
269:         funcs = [
270:             lambda x, y: x**2,
271:             lambda x, y: y**2,
272:             lambda x, y: x**2 - y**2,
273:             lambda x, y: x*y,
274:         ]
275: 
276:         for j, func in enumerate(funcs):
277:             self._check_accuracy(func, tol=1e-9, atol=0.22, rtol=0,
278:                                  err_msg="Function %d" % j)
279:             self._check_accuracy(func, tol=1e-9, atol=0.22, rtol=0,
280:                                  err_msg="Function %d" % j, rescale=True)
281: 
282:     def test_tri_input(self):
283:         # Test at single points
284:         x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
285:                      dtype=np.double)
286:         y = np.arange(x.shape[0], dtype=np.double)
287:         y = y - 3j*y
288: 
289:         tri = qhull.Delaunay(x)
290:         yi = interpnd.CloughTocher2DInterpolator(tri, y)(x)
291:         assert_almost_equal(y, yi)
292: 
293:     def test_tri_input_rescale(self):
294:         # Test at single points
295:         x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
296:                      dtype=np.double)
297:         y = np.arange(x.shape[0], dtype=np.double)
298:         y = y - 3j*y
299: 
300:         tri = qhull.Delaunay(x)
301:         try:
302:             interpnd.CloughTocher2DInterpolator(tri, y, rescale=True)(x)
303:         except ValueError as a:
304:             if str(a) != ("Rescaling is not supported when passing a "
305:                           "Delaunay triangulation as ``points``."):
306:                 raise
307:         except:
308:             raise
309: 
310:     def test_tripoints_input_rescale(self):
311:         # Test at single points
312:         x = np.array([(0,0), (-5,-5), (-5,5), (5, 5), (2.5, 3)],
313:                      dtype=np.double)
314:         y = np.arange(x.shape[0], dtype=np.double)
315:         y = y - 3j*y
316: 
317:         tri = qhull.Delaunay(x)
318:         yi = interpnd.CloughTocher2DInterpolator(tri.points, y)(x)
319:         yi_rescale = interpnd.CloughTocher2DInterpolator(tri.points, y, rescale=True)(x)
320:         assert_almost_equal(yi, yi_rescale)
321: 
322:     def test_dense(self):
323:         # Should be more accurate for dense meshes
324:         funcs = [
325:             lambda x, y: x**2,
326:             lambda x, y: y**2,
327:             lambda x, y: x**2 - y**2,
328:             lambda x, y: x*y,
329:             lambda x, y: np.cos(2*np.pi*x)*np.sin(2*np.pi*y)
330:         ]
331: 
332:         np.random.seed(4321)  # use a different seed than the check!
333:         grid = np.r_[np.array([(0,0), (0,1), (1,0), (1,1)], dtype=float),
334:                      np.random.rand(30*30, 2)]
335: 
336:         for j, func in enumerate(funcs):
337:             self._check_accuracy(func, x=grid, tol=1e-9, atol=5e-3, rtol=1e-2,
338:                                  err_msg="Function %d" % j)
339:             self._check_accuracy(func, x=grid, tol=1e-9, atol=5e-3, rtol=1e-2,
340:                                  err_msg="Function %d" % j, rescale=True)
341: 
342:     def test_wrong_ndim(self):
343:         x = np.random.randn(30, 3)
344:         y = np.random.randn(30)
345:         assert_raises(ValueError, interpnd.CloughTocher2DInterpolator, x, y)
346: 
347:     def test_pickle(self):
348:         # Test at single points
349:         np.random.seed(1234)
350:         x = np.random.rand(30, 2)
351:         y = np.random.rand(30) + 1j*np.random.rand(30)
352: 
353:         ip = interpnd.CloughTocher2DInterpolator(x, y)
354:         ip2 = pickle.loads(pickle.dumps(ip))
355: 
356:         assert_almost_equal(ip(0.5, 0.5), ip2(0.5, 0.5))
357: 

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
import_95578 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_95578) is not StypyTypeError):

    if (import_95578 != 'pyd_module'):
        __import__(import_95578)
        sys_modules_95579 = sys.modules[import_95578]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_95579.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_95578)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_equal, assert_allclose, assert_almost_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95580 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_95580) is not StypyTypeError):

    if (import_95580 != 'pyd_module'):
        __import__(import_95580)
        sys_modules_95581 = sys.modules[import_95580]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_95581.module_type_store, module_type_store, ['assert_equal', 'assert_allclose', 'assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_95581, sys_modules_95581.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose, assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose', 'assert_almost_equal'], [assert_equal, assert_allclose, assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_95580)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from pytest import assert_raises' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95582 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_95582) is not StypyTypeError):

    if (import_95582 != 'pyd_module'):
        __import__(import_95582)
        sys_modules_95583 = sys.modules[import_95582]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_95583.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_95583, sys_modules_95583.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_95582)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95584 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat')

if (type(import_95584) is not StypyTypeError):

    if (import_95584 != 'pyd_module'):
        __import__(import_95584)
        sys_modules_95585 = sys.modules[import_95584]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', sys_modules_95585.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_95585, sys_modules_95585.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', import_95584)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.interpolate.interpnd' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95586 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd')

if (type(import_95586) is not StypyTypeError):

    if (import_95586 != 'pyd_module'):
        __import__(import_95586)
        sys_modules_95587 = sys.modules[import_95586]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'interpnd', sys_modules_95587.module_type_store, module_type_store)
    else:
        import scipy.interpolate.interpnd as interpnd

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'interpnd', scipy.interpolate.interpnd, module_type_store)

else:
    # Assigning a type to the variable 'scipy.interpolate.interpnd' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate.interpnd', import_95586)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import scipy.spatial.qhull' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95588 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.spatial.qhull')

if (type(import_95588) is not StypyTypeError):

    if (import_95588 != 'pyd_module'):
        __import__(import_95588)
        sys_modules_95589 = sys.modules[import_95588]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'qhull', sys_modules_95589.module_type_store, module_type_store)
    else:
        import scipy.spatial.qhull as qhull

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'qhull', scipy.spatial.qhull, module_type_store)

else:
    # Assigning a type to the variable 'scipy.spatial.qhull' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.spatial.qhull', import_95588)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import pickle' statement (line 13)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'pickle', pickle, module_type_store)


@norecursion
def data_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'data_file'
    module_type_store = module_type_store.open_function_context('data_file', 16, 0, False)
    
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

    
    # Call to join(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to abspath(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to dirname(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of '__file__' (line 17)
    file___95599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 56), '__file__', False)
    # Processing the call keyword arguments (line 17)
    kwargs_95600 = {}
    # Getting the type of 'os' (line 17)
    os_95596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_95597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 40), os_95596, 'path')
    # Obtaining the member 'dirname' of a type (line 17)
    dirname_95598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 40), path_95597, 'dirname')
    # Calling dirname(args, kwargs) (line 17)
    dirname_call_result_95601 = invoke(stypy.reporting.localization.Localization(__file__, 17, 40), dirname_95598, *[file___95599], **kwargs_95600)
    
    # Processing the call keyword arguments (line 17)
    kwargs_95602 = {}
    # Getting the type of 'os' (line 17)
    os_95593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_95594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), os_95593, 'path')
    # Obtaining the member 'abspath' of a type (line 17)
    abspath_95595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 24), path_95594, 'abspath')
    # Calling abspath(args, kwargs) (line 17)
    abspath_call_result_95603 = invoke(stypy.reporting.localization.Localization(__file__, 17, 24), abspath_95595, *[dirname_call_result_95601], **kwargs_95602)
    
    str_95604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'data')
    # Getting the type of 'basename' (line 18)
    basename_95605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 32), 'basename', False)
    # Processing the call keyword arguments (line 17)
    kwargs_95606 = {}
    # Getting the type of 'os' (line 17)
    os_95590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 17)
    path_95591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), os_95590, 'path')
    # Obtaining the member 'join' of a type (line 17)
    join_95592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), path_95591, 'join')
    # Calling join(args, kwargs) (line 17)
    join_call_result_95607 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), join_95592, *[abspath_call_result_95603, str_95604, basename_95605], **kwargs_95606)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', join_call_result_95607)
    
    # ################# End of 'data_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'data_file' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_95608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_95608)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'data_file'
    return stypy_return_type_95608

# Assigning a type to the variable 'data_file' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'data_file', data_file)
# Declaration of the 'TestLinearNDInterpolation' class

class TestLinearNDInterpolation(object, ):

    @norecursion
    def test_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoketest'
        module_type_store = module_type_store.open_function_context('test_smoketest', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_smoketest')
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoketest(...)' code ##################

        
        # Assigning a Call to a Name (line 24):
        
        # Assigning a Call to a Name (line 24):
        
        # Call to array(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'list' (line 24)
        list_95611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 24)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_95612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        int_95613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), tuple_95612, int_95613)
        # Adding element type (line 24)
        int_95614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 23), tuple_95612, int_95614)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_95611, tuple_95612)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_95615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        float_95616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), tuple_95615, float_95616)
        # Adding element type (line 24)
        float_95617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 30), tuple_95615, float_95617)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_95611, tuple_95615)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_95618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        float_95619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 43), tuple_95618, float_95619)
        # Adding element type (line 24)
        float_95620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 43), tuple_95618, float_95620)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_95611, tuple_95618)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_95621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        float_95622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 55), tuple_95621, float_95622)
        # Adding element type (line 24)
        float_95623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 55), tuple_95621, float_95623)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_95611, tuple_95621)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_95624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        float_95625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 67), tuple_95624, float_95625)
        # Adding element type (line 24)
        float_95626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 67), tuple_95624, float_95626)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), list_95611, tuple_95624)
        
        # Processing the call keyword arguments (line 24)
        # Getting the type of 'np' (line 25)
        np_95627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 25)
        double_95628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 27), np_95627, 'double')
        keyword_95629 = double_95628
        kwargs_95630 = {'dtype': keyword_95629}
        # Getting the type of 'np' (line 24)
        np_95609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 24)
        array_95610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), np_95609, 'array')
        # Calling array(args, kwargs) (line 24)
        array_call_result_95631 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), array_95610, *[list_95611], **kwargs_95630)
        
        # Assigning a type to the variable 'x' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'x', array_call_result_95631)
        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to arange(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Obtaining the type of the subscript
        int_95634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'int')
        # Getting the type of 'x' (line 26)
        x_95635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 26)
        shape_95636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 22), x_95635, 'shape')
        # Obtaining the member '__getitem__' of a type (line 26)
        getitem___95637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 22), shape_95636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 26)
        subscript_call_result_95638 = invoke(stypy.reporting.localization.Localization(__file__, 26, 22), getitem___95637, int_95634)
        
        # Processing the call keyword arguments (line 26)
        # Getting the type of 'np' (line 26)
        np_95639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 26)
        double_95640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 40), np_95639, 'double')
        keyword_95641 = double_95640
        kwargs_95642 = {'dtype': keyword_95641}
        # Getting the type of 'np' (line 26)
        np_95632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 26)
        arange_95633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), np_95632, 'arange')
        # Calling arange(args, kwargs) (line 26)
        arange_call_result_95643 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), arange_95633, *[subscript_call_result_95638], **kwargs_95642)
        
        # Assigning a type to the variable 'y' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'y', arange_call_result_95643)
        
        # Assigning a Call to a Name (line 28):
        
        # Assigning a Call to a Name (line 28):
        
        # Call to (...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'x' (line 28)
        x_95650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_95651 = {}
        
        # Call to LinearNDInterpolator(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'x' (line 28)
        x_95646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'x', False)
        # Getting the type of 'y' (line 28)
        y_95647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'y', False)
        # Processing the call keyword arguments (line 28)
        kwargs_95648 = {}
        # Getting the type of 'interpnd' (line 28)
        interpnd_95644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 28)
        LinearNDInterpolator_95645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 13), interpnd_95644, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 28)
        LinearNDInterpolator_call_result_95649 = invoke(stypy.reporting.localization.Localization(__file__, 28, 13), LinearNDInterpolator_95645, *[x_95646, y_95647], **kwargs_95648)
        
        # Calling (args, kwargs) (line 28)
        _call_result_95652 = invoke(stypy.reporting.localization.Localization(__file__, 28, 13), LinearNDInterpolator_call_result_95649, *[x_95650], **kwargs_95651)
        
        # Assigning a type to the variable 'yi' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'yi', _call_result_95652)
        
        # Call to assert_almost_equal(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'y' (line 29)
        y_95654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'y', False)
        # Getting the type of 'yi' (line 29)
        yi_95655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'yi', False)
        # Processing the call keyword arguments (line 29)
        kwargs_95656 = {}
        # Getting the type of 'assert_almost_equal' (line 29)
        assert_almost_equal_95653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 29)
        assert_almost_equal_call_result_95657 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert_almost_equal_95653, *[y_95654, yi_95655], **kwargs_95656)
        
        
        # ################# End of 'test_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_95658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoketest'
        return stypy_return_type_95658


    @norecursion
    def test_smoketest_alternate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoketest_alternate'
        module_type_store = module_type_store.open_function_context('test_smoketest_alternate', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_smoketest_alternate')
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_smoketest_alternate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_smoketest_alternate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoketest_alternate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoketest_alternate(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to array(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_95661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_95662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        int_95663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 23), tuple_95662, int_95663)
        # Adding element type (line 33)
        int_95664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 23), tuple_95662, int_95664)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_95661, tuple_95662)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_95665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        float_95666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 30), tuple_95665, float_95666)
        # Adding element type (line 33)
        float_95667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 30), tuple_95665, float_95667)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_95661, tuple_95665)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_95668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        float_95669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 43), tuple_95668, float_95669)
        # Adding element type (line 33)
        float_95670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 43), tuple_95668, float_95670)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_95661, tuple_95668)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_95671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        float_95672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 55), tuple_95671, float_95672)
        # Adding element type (line 33)
        float_95673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 55), tuple_95671, float_95673)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_95661, tuple_95671)
        # Adding element type (line 33)
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_95674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        float_95675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 67), tuple_95674, float_95675)
        # Adding element type (line 33)
        float_95676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 67), tuple_95674, float_95676)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_95661, tuple_95674)
        
        # Processing the call keyword arguments (line 33)
        # Getting the type of 'np' (line 34)
        np_95677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 34)
        double_95678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 27), np_95677, 'double')
        keyword_95679 = double_95678
        kwargs_95680 = {'dtype': keyword_95679}
        # Getting the type of 'np' (line 33)
        np_95659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 33)
        array_95660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), np_95659, 'array')
        # Calling array(args, kwargs) (line 33)
        array_call_result_95681 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), array_95660, *[list_95661], **kwargs_95680)
        
        # Assigning a type to the variable 'x' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'x', array_call_result_95681)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to arange(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Obtaining the type of the subscript
        int_95684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'int')
        # Getting the type of 'x' (line 35)
        x_95685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 35)
        shape_95686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), x_95685, 'shape')
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___95687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), shape_95686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_95688 = invoke(stypy.reporting.localization.Localization(__file__, 35, 22), getitem___95687, int_95684)
        
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'np' (line 35)
        np_95689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 35)
        double_95690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 40), np_95689, 'double')
        keyword_95691 = double_95690
        kwargs_95692 = {'dtype': keyword_95691}
        # Getting the type of 'np' (line 35)
        np_95682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 35)
        arange_95683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), np_95682, 'arange')
        # Calling arange(args, kwargs) (line 35)
        arange_call_result_95693 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), arange_95683, *[subscript_call_result_95688], **kwargs_95692)
        
        # Assigning a type to the variable 'y' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'y', arange_call_result_95693)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to (...): (line 37)
        # Processing the call arguments (line 37)
        
        # Obtaining the type of the subscript
        slice_95710 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 64), None, None, None)
        int_95711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 68), 'int')
        # Getting the type of 'x' (line 37)
        x_95712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 64), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___95713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 64), x_95712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_95714 = invoke(stypy.reporting.localization.Localization(__file__, 37, 64), getitem___95713, (slice_95710, int_95711))
        
        
        # Obtaining the type of the subscript
        slice_95715 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 72), None, None, None)
        int_95716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 76), 'int')
        # Getting the type of 'x' (line 37)
        x_95717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 72), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___95718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 72), x_95717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_95719 = invoke(stypy.reporting.localization.Localization(__file__, 37, 72), getitem___95718, (slice_95715, int_95716))
        
        # Processing the call keyword arguments (line 37)
        kwargs_95720 = {}
        
        # Call to LinearNDInterpolator(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Obtaining an instance of the builtin type 'tuple' (line 37)
        tuple_95696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 37)
        # Adding element type (line 37)
        
        # Obtaining the type of the subscript
        slice_95697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 44), None, None, None)
        int_95698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 48), 'int')
        # Getting the type of 'x' (line 37)
        x_95699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 44), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___95700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 44), x_95699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_95701 = invoke(stypy.reporting.localization.Localization(__file__, 37, 44), getitem___95700, (slice_95697, int_95698))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 44), tuple_95696, subscript_call_result_95701)
        # Adding element type (line 37)
        
        # Obtaining the type of the subscript
        slice_95702 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 52), None, None, None)
        int_95703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'int')
        # Getting the type of 'x' (line 37)
        x_95704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___95705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 52), x_95704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
        subscript_call_result_95706 = invoke(stypy.reporting.localization.Localization(__file__, 37, 52), getitem___95705, (slice_95702, int_95703))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 44), tuple_95696, subscript_call_result_95706)
        
        # Getting the type of 'y' (line 37)
        y_95707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 61), 'y', False)
        # Processing the call keyword arguments (line 37)
        kwargs_95708 = {}
        # Getting the type of 'interpnd' (line 37)
        interpnd_95694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 37)
        LinearNDInterpolator_95695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), interpnd_95694, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 37)
        LinearNDInterpolator_call_result_95709 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), LinearNDInterpolator_95695, *[tuple_95696, y_95707], **kwargs_95708)
        
        # Calling (args, kwargs) (line 37)
        _call_result_95721 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), LinearNDInterpolator_call_result_95709, *[subscript_call_result_95714, subscript_call_result_95719], **kwargs_95720)
        
        # Assigning a type to the variable 'yi' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'yi', _call_result_95721)
        
        # Call to assert_almost_equal(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'y' (line 38)
        y_95723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'y', False)
        # Getting the type of 'yi' (line 38)
        yi_95724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'yi', False)
        # Processing the call keyword arguments (line 38)
        kwargs_95725 = {}
        # Getting the type of 'assert_almost_equal' (line 38)
        assert_almost_equal_95722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 38)
        assert_almost_equal_call_result_95726 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_almost_equal_95722, *[y_95723, yi_95724], **kwargs_95725)
        
        
        # ################# End of 'test_smoketest_alternate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoketest_alternate' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_95727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoketest_alternate'
        return stypy_return_type_95727


    @norecursion
    def test_complex_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex_smoketest'
        module_type_store = module_type_store.open_function_context('test_complex_smoketest', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_complex_smoketest')
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_complex_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_complex_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex_smoketest(...)' code ##################

        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to array(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_95730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_95731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        int_95732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 23), tuple_95731, int_95732)
        # Adding element type (line 42)
        int_95733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 23), tuple_95731, int_95733)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_95730, tuple_95731)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_95734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        float_95735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 30), tuple_95734, float_95735)
        # Adding element type (line 42)
        float_95736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 30), tuple_95734, float_95736)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_95730, tuple_95734)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_95737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        float_95738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 43), tuple_95737, float_95738)
        # Adding element type (line 42)
        float_95739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 43), tuple_95737, float_95739)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_95730, tuple_95737)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_95740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        float_95741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 55), tuple_95740, float_95741)
        # Adding element type (line 42)
        float_95742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 55), tuple_95740, float_95742)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_95730, tuple_95740)
        # Adding element type (line 42)
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_95743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        float_95744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 67), tuple_95743, float_95744)
        # Adding element type (line 42)
        float_95745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 67), tuple_95743, float_95745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), list_95730, tuple_95743)
        
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'np' (line 43)
        np_95746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 43)
        double_95747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), np_95746, 'double')
        keyword_95748 = double_95747
        kwargs_95749 = {'dtype': keyword_95748}
        # Getting the type of 'np' (line 42)
        np_95728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 42)
        array_95729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), np_95728, 'array')
        # Calling array(args, kwargs) (line 42)
        array_call_result_95750 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), array_95729, *[list_95730], **kwargs_95749)
        
        # Assigning a type to the variable 'x' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'x', array_call_result_95750)
        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to arange(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Obtaining the type of the subscript
        int_95753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'int')
        # Getting the type of 'x' (line 44)
        x_95754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 44)
        shape_95755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), x_95754, 'shape')
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___95756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), shape_95755, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_95757 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), getitem___95756, int_95753)
        
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'np' (line 44)
        np_95758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 44)
        double_95759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 40), np_95758, 'double')
        keyword_95760 = double_95759
        kwargs_95761 = {'dtype': keyword_95760}
        # Getting the type of 'np' (line 44)
        np_95751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 44)
        arange_95752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), np_95751, 'arange')
        # Calling arange(args, kwargs) (line 44)
        arange_call_result_95762 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), arange_95752, *[subscript_call_result_95757], **kwargs_95761)
        
        # Assigning a type to the variable 'y' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'y', arange_call_result_95762)
        
        # Assigning a BinOp to a Name (line 45):
        
        # Assigning a BinOp to a Name (line 45):
        # Getting the type of 'y' (line 45)
        y_95763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'y')
        complex_95764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'complex')
        # Getting the type of 'y' (line 45)
        y_95765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'y')
        # Applying the binary operator '*' (line 45)
        result_mul_95766 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 16), '*', complex_95764, y_95765)
        
        # Applying the binary operator '-' (line 45)
        result_sub_95767 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '-', y_95763, result_mul_95766)
        
        # Assigning a type to the variable 'y' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'y', result_sub_95767)
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to (...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'x' (line 47)
        x_95774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 49), 'x', False)
        # Processing the call keyword arguments (line 47)
        kwargs_95775 = {}
        
        # Call to LinearNDInterpolator(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'x' (line 47)
        x_95770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 43), 'x', False)
        # Getting the type of 'y' (line 47)
        y_95771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 46), 'y', False)
        # Processing the call keyword arguments (line 47)
        kwargs_95772 = {}
        # Getting the type of 'interpnd' (line 47)
        interpnd_95768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 47)
        LinearNDInterpolator_95769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 13), interpnd_95768, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 47)
        LinearNDInterpolator_call_result_95773 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), LinearNDInterpolator_95769, *[x_95770, y_95771], **kwargs_95772)
        
        # Calling (args, kwargs) (line 47)
        _call_result_95776 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), LinearNDInterpolator_call_result_95773, *[x_95774], **kwargs_95775)
        
        # Assigning a type to the variable 'yi' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'yi', _call_result_95776)
        
        # Call to assert_almost_equal(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'y' (line 48)
        y_95778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'y', False)
        # Getting the type of 'yi' (line 48)
        yi_95779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'yi', False)
        # Processing the call keyword arguments (line 48)
        kwargs_95780 = {}
        # Getting the type of 'assert_almost_equal' (line 48)
        assert_almost_equal_95777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 48)
        assert_almost_equal_call_result_95781 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_almost_equal_95777, *[y_95778, yi_95779], **kwargs_95780)
        
        
        # ################# End of 'test_complex_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_95782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex_smoketest'
        return stypy_return_type_95782


    @norecursion
    def test_tri_input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tri_input'
        module_type_store = module_type_store.open_function_context('test_tri_input', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_tri_input')
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_tri_input.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_tri_input', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tri_input', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tri_input(...)' code ##################

        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to array(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining an instance of the builtin type 'list' (line 52)
        list_95785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 52)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_95786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        int_95787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 23), tuple_95786, int_95787)
        # Adding element type (line 52)
        int_95788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 23), tuple_95786, int_95788)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_95785, tuple_95786)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_95789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        float_95790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 30), tuple_95789, float_95790)
        # Adding element type (line 52)
        float_95791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 30), tuple_95789, float_95791)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_95785, tuple_95789)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_95792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        float_95793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 43), tuple_95792, float_95793)
        # Adding element type (line 52)
        float_95794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 43), tuple_95792, float_95794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_95785, tuple_95792)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_95795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        float_95796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 55), tuple_95795, float_95796)
        # Adding element type (line 52)
        float_95797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 55), tuple_95795, float_95797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_95785, tuple_95795)
        # Adding element type (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_95798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        float_95799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 67), tuple_95798, float_95799)
        # Adding element type (line 52)
        float_95800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 67), tuple_95798, float_95800)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 21), list_95785, tuple_95798)
        
        # Processing the call keyword arguments (line 52)
        # Getting the type of 'np' (line 53)
        np_95801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 53)
        double_95802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 27), np_95801, 'double')
        keyword_95803 = double_95802
        kwargs_95804 = {'dtype': keyword_95803}
        # Getting the type of 'np' (line 52)
        np_95783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 52)
        array_95784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), np_95783, 'array')
        # Calling array(args, kwargs) (line 52)
        array_call_result_95805 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), array_95784, *[list_95785], **kwargs_95804)
        
        # Assigning a type to the variable 'x' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'x', array_call_result_95805)
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to arange(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining the type of the subscript
        int_95808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'int')
        # Getting the type of 'x' (line 54)
        x_95809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 54)
        shape_95810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 22), x_95809, 'shape')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___95811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 22), shape_95810, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_95812 = invoke(stypy.reporting.localization.Localization(__file__, 54, 22), getitem___95811, int_95808)
        
        # Processing the call keyword arguments (line 54)
        # Getting the type of 'np' (line 54)
        np_95813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 54)
        double_95814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 40), np_95813, 'double')
        keyword_95815 = double_95814
        kwargs_95816 = {'dtype': keyword_95815}
        # Getting the type of 'np' (line 54)
        np_95806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 54)
        arange_95807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), np_95806, 'arange')
        # Calling arange(args, kwargs) (line 54)
        arange_call_result_95817 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), arange_95807, *[subscript_call_result_95812], **kwargs_95816)
        
        # Assigning a type to the variable 'y' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'y', arange_call_result_95817)
        
        # Assigning a BinOp to a Name (line 55):
        
        # Assigning a BinOp to a Name (line 55):
        # Getting the type of 'y' (line 55)
        y_95818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'y')
        complex_95819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'complex')
        # Getting the type of 'y' (line 55)
        y_95820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'y')
        # Applying the binary operator '*' (line 55)
        result_mul_95821 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 16), '*', complex_95819, y_95820)
        
        # Applying the binary operator '-' (line 55)
        result_sub_95822 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 12), '-', y_95818, result_mul_95821)
        
        # Assigning a type to the variable 'y' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'y', result_sub_95822)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to Delaunay(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'x' (line 57)
        x_95825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'x', False)
        # Processing the call keyword arguments (line 57)
        kwargs_95826 = {}
        # Getting the type of 'qhull' (line 57)
        qhull_95823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 57)
        Delaunay_95824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), qhull_95823, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 57)
        Delaunay_call_result_95827 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), Delaunay_95824, *[x_95825], **kwargs_95826)
        
        # Assigning a type to the variable 'tri' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'tri', Delaunay_call_result_95827)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to (...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'x' (line 58)
        x_95834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 51), 'x', False)
        # Processing the call keyword arguments (line 58)
        kwargs_95835 = {}
        
        # Call to LinearNDInterpolator(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'tri' (line 58)
        tri_95830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'tri', False)
        # Getting the type of 'y' (line 58)
        y_95831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 48), 'y', False)
        # Processing the call keyword arguments (line 58)
        kwargs_95832 = {}
        # Getting the type of 'interpnd' (line 58)
        interpnd_95828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 58)
        LinearNDInterpolator_95829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 13), interpnd_95828, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 58)
        LinearNDInterpolator_call_result_95833 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), LinearNDInterpolator_95829, *[tri_95830, y_95831], **kwargs_95832)
        
        # Calling (args, kwargs) (line 58)
        _call_result_95836 = invoke(stypy.reporting.localization.Localization(__file__, 58, 13), LinearNDInterpolator_call_result_95833, *[x_95834], **kwargs_95835)
        
        # Assigning a type to the variable 'yi' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'yi', _call_result_95836)
        
        # Call to assert_almost_equal(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'y' (line 59)
        y_95838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'y', False)
        # Getting the type of 'yi' (line 59)
        yi_95839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'yi', False)
        # Processing the call keyword arguments (line 59)
        kwargs_95840 = {}
        # Getting the type of 'assert_almost_equal' (line 59)
        assert_almost_equal_95837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 59)
        assert_almost_equal_call_result_95841 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_almost_equal_95837, *[y_95838, yi_95839], **kwargs_95840)
        
        
        # ################# End of 'test_tri_input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tri_input' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_95842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tri_input'
        return stypy_return_type_95842


    @norecursion
    def test_square(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square'
        module_type_store = module_type_store.open_function_context('test_square', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_square')
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_square.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_square', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square(...)' code ##################

        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_95845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_95846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        int_95847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), tuple_95846, int_95847)
        # Adding element type (line 65)
        int_95848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 28), tuple_95846, int_95848)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_95845, tuple_95846)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_95849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        int_95850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 35), tuple_95849, int_95850)
        # Adding element type (line 65)
        int_95851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 35), tuple_95849, int_95851)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_95845, tuple_95849)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_95852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        int_95853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 42), tuple_95852, int_95853)
        # Adding element type (line 65)
        int_95854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 42), tuple_95852, int_95854)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_95845, tuple_95852)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'tuple' (line 65)
        tuple_95855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 65)
        # Adding element type (line 65)
        int_95856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 49), tuple_95855, int_95856)
        # Adding element type (line 65)
        int_95857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 49), tuple_95855, int_95857)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 26), list_95845, tuple_95855)
        
        # Processing the call keyword arguments (line 65)
        # Getting the type of 'np' (line 65)
        np_95858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 62), 'np', False)
        # Obtaining the member 'double' of a type (line 65)
        double_95859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 62), np_95858, 'double')
        keyword_95860 = double_95859
        kwargs_95861 = {'dtype': keyword_95860}
        # Getting the type of 'np' (line 65)
        np_95843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_95844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 17), np_95843, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_95862 = invoke(stypy.reporting.localization.Localization(__file__, 65, 17), array_95844, *[list_95845], **kwargs_95861)
        
        # Assigning a type to the variable 'points' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'points', array_call_result_95862)
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to array(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_95865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_95866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), list_95865, float_95866)
        # Adding element type (line 66)
        float_95867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), list_95865, float_95867)
        # Adding element type (line 66)
        float_95868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), list_95865, float_95868)
        # Adding element type (line 66)
        float_95869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 26), list_95865, float_95869)
        
        # Processing the call keyword arguments (line 66)
        # Getting the type of 'np' (line 66)
        np_95870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'np', False)
        # Obtaining the member 'double' of a type (line 66)
        double_95871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 51), np_95870, 'double')
        keyword_95872 = double_95871
        kwargs_95873 = {'dtype': keyword_95872}
        # Getting the type of 'np' (line 66)
        np_95863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 66)
        array_95864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 17), np_95863, 'array')
        # Calling array(args, kwargs) (line 66)
        array_call_result_95874 = invoke(stypy.reporting.localization.Localization(__file__, 66, 17), array_95864, *[list_95865], **kwargs_95873)
        
        # Assigning a type to the variable 'values' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'values', array_call_result_95874)

        @norecursion
        def ip(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'ip'
            module_type_store = module_type_store.open_function_context('ip', 75, 8, False)
            
            # Passed parameters checking function
            ip.stypy_localization = localization
            ip.stypy_type_of_self = None
            ip.stypy_type_store = module_type_store
            ip.stypy_function_name = 'ip'
            ip.stypy_param_names_list = ['x', 'y']
            ip.stypy_varargs_param_name = None
            ip.stypy_kwargs_param_name = None
            ip.stypy_call_defaults = defaults
            ip.stypy_call_varargs = varargs
            ip.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'ip', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'ip', localization, ['x', 'y'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'ip(...)' code ##################

            
            # Assigning a Compare to a Name (line 76):
            
            # Assigning a Compare to a Name (line 76):
            
            # Getting the type of 'x' (line 76)
            x_95875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 18), 'x')
            # Getting the type of 'y' (line 76)
            y_95876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'y')
            # Applying the binary operator '+' (line 76)
            result_add_95877 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 18), '+', x_95875, y_95876)
            
            int_95878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 27), 'int')
            # Applying the binary operator '<=' (line 76)
            result_le_95879 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 18), '<=', result_add_95877, int_95878)
            
            # Assigning a type to the variable 't1' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 't1', result_le_95879)
            
            # Assigning a UnaryOp to a Name (line 77):
            
            # Assigning a UnaryOp to a Name (line 77):
            
            # Getting the type of 't1' (line 77)
            t1_95880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 't1')
            # Applying the '~' unary operator (line 77)
            result_inv_95881 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 17), '~', t1_95880)
            
            # Assigning a type to the variable 't2' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 't2', result_inv_95881)
            
            # Assigning a Subscript to a Name (line 79):
            
            # Assigning a Subscript to a Name (line 79):
            
            # Obtaining the type of the subscript
            # Getting the type of 't1' (line 79)
            t1_95882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 't1')
            # Getting the type of 'x' (line 79)
            x_95883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'x')
            # Obtaining the member '__getitem__' of a type (line 79)
            getitem___95884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 17), x_95883, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 79)
            subscript_call_result_95885 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), getitem___95884, t1_95882)
            
            # Assigning a type to the variable 'x1' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'x1', subscript_call_result_95885)
            
            # Assigning a Subscript to a Name (line 80):
            
            # Assigning a Subscript to a Name (line 80):
            
            # Obtaining the type of the subscript
            # Getting the type of 't1' (line 80)
            t1_95886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 't1')
            # Getting the type of 'y' (line 80)
            y_95887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'y')
            # Obtaining the member '__getitem__' of a type (line 80)
            getitem___95888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), y_95887, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
            subscript_call_result_95889 = invoke(stypy.reporting.localization.Localization(__file__, 80, 17), getitem___95888, t1_95886)
            
            # Assigning a type to the variable 'y1' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'y1', subscript_call_result_95889)
            
            # Assigning a Subscript to a Name (line 82):
            
            # Assigning a Subscript to a Name (line 82):
            
            # Obtaining the type of the subscript
            # Getting the type of 't2' (line 82)
            t2_95890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 't2')
            # Getting the type of 'x' (line 82)
            x_95891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'x')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___95892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), x_95891, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_95893 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), getitem___95892, t2_95890)
            
            # Assigning a type to the variable 'x2' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'x2', subscript_call_result_95893)
            
            # Assigning a Subscript to a Name (line 83):
            
            # Assigning a Subscript to a Name (line 83):
            
            # Obtaining the type of the subscript
            # Getting the type of 't2' (line 83)
            t2_95894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 't2')
            # Getting the type of 'y' (line 83)
            y_95895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'y')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___95896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), y_95895, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_95897 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), getitem___95896, t2_95894)
            
            # Assigning a type to the variable 'y2' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'y2', subscript_call_result_95897)
            
            # Assigning a BinOp to a Name (line 85):
            
            # Assigning a BinOp to a Name (line 85):
            int_95898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'int')
            # Getting the type of 'x' (line 85)
            x_95899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'x')
            # Applying the binary operator '*' (line 85)
            result_mul_95900 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 16), '*', int_95898, x_95899)
            
            # Assigning a type to the variable 'z' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'z', result_mul_95900)
            
            # Assigning a BinOp to a Subscript (line 87):
            
            # Assigning a BinOp to a Subscript (line 87):
            
            # Obtaining the type of the subscript
            int_95901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'int')
            # Getting the type of 'values' (line 87)
            values_95902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'values')
            # Obtaining the member '__getitem__' of a type (line 87)
            getitem___95903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 21), values_95902, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 87)
            subscript_call_result_95904 = invoke(stypy.reporting.localization.Localization(__file__, 87, 21), getitem___95903, int_95901)
            
            int_95905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'int')
            # Getting the type of 'x1' (line 87)
            x1_95906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'x1')
            # Applying the binary operator '-' (line 87)
            result_sub_95907 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 32), '-', int_95905, x1_95906)
            
            # Getting the type of 'y1' (line 87)
            y1_95908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'y1')
            # Applying the binary operator '-' (line 87)
            result_sub_95909 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 39), '-', result_sub_95907, y1_95908)
            
            # Applying the binary operator '*' (line 87)
            result_mul_95910 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 21), '*', subscript_call_result_95904, result_sub_95909)
            
            
            # Obtaining the type of the subscript
            int_95911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
            # Getting the type of 'values' (line 88)
            values_95912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'values')
            # Obtaining the member '__getitem__' of a type (line 88)
            getitem___95913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 23), values_95912, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 88)
            subscript_call_result_95914 = invoke(stypy.reporting.localization.Localization(__file__, 88, 23), getitem___95913, int_95911)
            
            # Getting the type of 'y1' (line 88)
            y1_95915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 33), 'y1')
            # Applying the binary operator '*' (line 88)
            result_mul_95916 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 23), '*', subscript_call_result_95914, y1_95915)
            
            # Applying the binary operator '+' (line 87)
            result_add_95917 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 21), '+', result_mul_95910, result_mul_95916)
            
            
            # Obtaining the type of the subscript
            int_95918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'int')
            # Getting the type of 'values' (line 89)
            values_95919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'values')
            # Obtaining the member '__getitem__' of a type (line 89)
            getitem___95920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), values_95919, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 89)
            subscript_call_result_95921 = invoke(stypy.reporting.localization.Localization(__file__, 89, 23), getitem___95920, int_95918)
            
            # Getting the type of 'x1' (line 89)
            x1_95922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'x1')
            # Applying the binary operator '*' (line 89)
            result_mul_95923 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 23), '*', subscript_call_result_95921, x1_95922)
            
            # Applying the binary operator '+' (line 89)
            result_add_95924 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 21), '+', result_add_95917, result_mul_95923)
            
            # Getting the type of 'z' (line 87)
            z_95925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'z')
            # Getting the type of 't1' (line 87)
            t1_95926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 't1')
            # Storing an element on a container (line 87)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 12), z_95925, (t1_95926, result_add_95924))
            
            # Assigning a BinOp to a Subscript (line 91):
            
            # Assigning a BinOp to a Subscript (line 91):
            
            # Obtaining the type of the subscript
            int_95927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'int')
            # Getting the type of 'values' (line 91)
            values_95928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'values')
            # Obtaining the member '__getitem__' of a type (line 91)
            getitem___95929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), values_95928, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 91)
            subscript_call_result_95930 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), getitem___95929, int_95927)
            
            # Getting the type of 'x2' (line 91)
            x2_95931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'x2')
            # Getting the type of 'y2' (line 91)
            y2_95932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'y2')
            # Applying the binary operator '+' (line 91)
            result_add_95933 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 32), '+', x2_95931, y2_95932)
            
            int_95934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 42), 'int')
            # Applying the binary operator '-' (line 91)
            result_sub_95935 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 40), '-', result_add_95933, int_95934)
            
            # Applying the binary operator '*' (line 91)
            result_mul_95936 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '*', subscript_call_result_95930, result_sub_95935)
            
            
            # Obtaining the type of the subscript
            int_95937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 30), 'int')
            # Getting the type of 'values' (line 92)
            values_95938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'values')
            # Obtaining the member '__getitem__' of a type (line 92)
            getitem___95939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 23), values_95938, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 92)
            subscript_call_result_95940 = invoke(stypy.reporting.localization.Localization(__file__, 92, 23), getitem___95939, int_95937)
            
            int_95941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 34), 'int')
            # Getting the type of 'x2' (line 92)
            x2_95942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'x2')
            # Applying the binary operator '-' (line 92)
            result_sub_95943 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 34), '-', int_95941, x2_95942)
            
            # Applying the binary operator '*' (line 92)
            result_mul_95944 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 23), '*', subscript_call_result_95940, result_sub_95943)
            
            # Applying the binary operator '+' (line 91)
            result_add_95945 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '+', result_mul_95936, result_mul_95944)
            
            
            # Obtaining the type of the subscript
            int_95946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 30), 'int')
            # Getting the type of 'values' (line 93)
            values_95947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 23), 'values')
            # Obtaining the member '__getitem__' of a type (line 93)
            getitem___95948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 23), values_95947, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 93)
            subscript_call_result_95949 = invoke(stypy.reporting.localization.Localization(__file__, 93, 23), getitem___95948, int_95946)
            
            int_95950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'int')
            # Getting the type of 'y2' (line 93)
            y2_95951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'y2')
            # Applying the binary operator '-' (line 93)
            result_sub_95952 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '-', int_95950, y2_95951)
            
            # Applying the binary operator '*' (line 93)
            result_mul_95953 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 23), '*', subscript_call_result_95949, result_sub_95952)
            
            # Applying the binary operator '+' (line 93)
            result_add_95954 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 21), '+', result_add_95945, result_mul_95953)
            
            # Getting the type of 'z' (line 91)
            z_95955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'z')
            # Getting the type of 't2' (line 91)
            t2_95956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 't2')
            # Storing an element on a container (line 91)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 12), z_95955, (t2_95956, result_add_95954))
            # Getting the type of 'z' (line 94)
            z_95957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'z')
            # Assigning a type to the variable 'stypy_return_type' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', z_95957)
            
            # ################# End of 'ip(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'ip' in the type store
            # Getting the type of 'stypy_return_type' (line 75)
            stypy_return_type_95958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_95958)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'ip'
            return stypy_return_type_95958

        # Assigning a type to the variable 'ip' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'ip', ip)
        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_95959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining the type of the subscript
        slice_95962 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 37), None, None, None)
        # Getting the type of 'None' (line 96)
        None_95963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 61), 'None', False)
        
        # Call to linspace(...): (line 96)
        # Processing the call arguments (line 96)
        int_95966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 49), 'int')
        int_95967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'int')
        int_95968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 55), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_95969 = {}
        # Getting the type of 'np' (line 96)
        np_95964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 96)
        linspace_95965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 37), np_95964, 'linspace')
        # Calling linspace(args, kwargs) (line 96)
        linspace_call_result_95970 = invoke(stypy.reporting.localization.Localization(__file__, 96, 37), linspace_95965, *[int_95966, int_95967, int_95968], **kwargs_95969)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___95971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 37), linspace_call_result_95970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_95972 = invoke(stypy.reporting.localization.Localization(__file__, 96, 37), getitem___95971, (slice_95962, None_95963))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 97)
        None_95973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 59), 'None', False)
        slice_95974 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 37), None, None, None)
        
        # Call to linspace(...): (line 97)
        # Processing the call arguments (line 97)
        int_95977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 49), 'int')
        int_95978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 52), 'int')
        int_95979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_95980 = {}
        # Getting the type of 'np' (line 97)
        np_95975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 97)
        linspace_95976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 37), np_95975, 'linspace')
        # Calling linspace(args, kwargs) (line 97)
        linspace_call_result_95981 = invoke(stypy.reporting.localization.Localization(__file__, 97, 37), linspace_95976, *[int_95977, int_95978, int_95979], **kwargs_95980)
        
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___95982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 37), linspace_call_result_95981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_95983 = invoke(stypy.reporting.localization.Localization(__file__, 97, 37), getitem___95982, (None_95973, slice_95974))
        
        # Processing the call keyword arguments (line 96)
        kwargs_95984 = {}
        # Getting the type of 'np' (line 96)
        np_95960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 96)
        broadcast_arrays_95961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), np_95960, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 96)
        broadcast_arrays_call_result_95985 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), broadcast_arrays_95961, *[subscript_call_result_95972, subscript_call_result_95983], **kwargs_95984)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___95986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), broadcast_arrays_call_result_95985, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_95987 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___95986, int_95959)
        
        # Assigning a type to the variable 'tuple_var_assignment_95574' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_95574', subscript_call_result_95987)
        
        # Assigning a Subscript to a Name (line 96):
        
        # Obtaining the type of the subscript
        int_95988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining the type of the subscript
        slice_95991 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 96, 37), None, None, None)
        # Getting the type of 'None' (line 96)
        None_95992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 61), 'None', False)
        
        # Call to linspace(...): (line 96)
        # Processing the call arguments (line 96)
        int_95995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 49), 'int')
        int_95996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 52), 'int')
        int_95997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 55), 'int')
        # Processing the call keyword arguments (line 96)
        kwargs_95998 = {}
        # Getting the type of 'np' (line 96)
        np_95993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 96)
        linspace_95994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 37), np_95993, 'linspace')
        # Calling linspace(args, kwargs) (line 96)
        linspace_call_result_95999 = invoke(stypy.reporting.localization.Localization(__file__, 96, 37), linspace_95994, *[int_95995, int_95996, int_95997], **kwargs_95998)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___96000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 37), linspace_call_result_95999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_96001 = invoke(stypy.reporting.localization.Localization(__file__, 96, 37), getitem___96000, (slice_95991, None_95992))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 97)
        None_96002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 59), 'None', False)
        slice_96003 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 97, 37), None, None, None)
        
        # Call to linspace(...): (line 97)
        # Processing the call arguments (line 97)
        int_96006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 49), 'int')
        int_96007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 52), 'int')
        int_96008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_96009 = {}
        # Getting the type of 'np' (line 97)
        np_96004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 97)
        linspace_96005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 37), np_96004, 'linspace')
        # Calling linspace(args, kwargs) (line 97)
        linspace_call_result_96010 = invoke(stypy.reporting.localization.Localization(__file__, 97, 37), linspace_96005, *[int_96006, int_96007, int_96008], **kwargs_96009)
        
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___96011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 37), linspace_call_result_96010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_96012 = invoke(stypy.reporting.localization.Localization(__file__, 97, 37), getitem___96011, (None_96002, slice_96003))
        
        # Processing the call keyword arguments (line 96)
        kwargs_96013 = {}
        # Getting the type of 'np' (line 96)
        np_95989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 96)
        broadcast_arrays_95990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), np_95989, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 96)
        broadcast_arrays_call_result_96014 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), broadcast_arrays_95990, *[subscript_call_result_96001, subscript_call_result_96012], **kwargs_96013)
        
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___96015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), broadcast_arrays_call_result_96014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_96016 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___96015, int_95988)
        
        # Assigning a type to the variable 'tuple_var_assignment_95575' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_95575', subscript_call_result_96016)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_95574' (line 96)
        tuple_var_assignment_95574_96017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_95574')
        # Assigning a type to the variable 'xx' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'xx', tuple_var_assignment_95574_96017)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'tuple_var_assignment_95575' (line 96)
        tuple_var_assignment_95575_96018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_95575')
        # Assigning a type to the variable 'yy' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'yy', tuple_var_assignment_95575_96018)
        
        # Assigning a Call to a Name (line 98):
        
        # Assigning a Call to a Name (line 98):
        
        # Call to ravel(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_96021 = {}
        # Getting the type of 'xx' (line 98)
        xx_96019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'xx', False)
        # Obtaining the member 'ravel' of a type (line 98)
        ravel_96020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), xx_96019, 'ravel')
        # Calling ravel(args, kwargs) (line 98)
        ravel_call_result_96022 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), ravel_96020, *[], **kwargs_96021)
        
        # Assigning a type to the variable 'xx' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'xx', ravel_call_result_96022)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to ravel(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_96025 = {}
        # Getting the type of 'yy' (line 99)
        yy_96023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'yy', False)
        # Obtaining the member 'ravel' of a type (line 99)
        ravel_96024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), yy_96023, 'ravel')
        # Calling ravel(args, kwargs) (line 99)
        ravel_call_result_96026 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), ravel_96024, *[], **kwargs_96025)
        
        # Assigning a type to the variable 'yy' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'yy', ravel_call_result_96026)
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to copy(...): (line 101)
        # Processing the call keyword arguments (line 101)
        kwargs_96036 = {}
        
        # Call to array(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_96029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'xx' (line 101)
        xx_96030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'xx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_96029, xx_96030)
        # Adding element type (line 101)
        # Getting the type of 'yy' (line 101)
        yy_96031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'yy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 22), list_96029, yy_96031)
        
        # Processing the call keyword arguments (line 101)
        kwargs_96032 = {}
        # Getting the type of 'np' (line 101)
        np_96027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 101)
        array_96028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), np_96027, 'array')
        # Calling array(args, kwargs) (line 101)
        array_call_result_96033 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), array_96028, *[list_96029], **kwargs_96032)
        
        # Obtaining the member 'T' of a type (line 101)
        T_96034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), array_call_result_96033, 'T')
        # Obtaining the member 'copy' of a type (line 101)
        copy_96035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 13), T_96034, 'copy')
        # Calling copy(args, kwargs) (line 101)
        copy_call_result_96037 = invoke(stypy.reporting.localization.Localization(__file__, 101, 13), copy_96035, *[], **kwargs_96036)
        
        # Assigning a type to the variable 'xi' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'xi', copy_call_result_96037)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to (...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'xi' (line 102)
        xi_96044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 59), 'xi', False)
        # Processing the call keyword arguments (line 102)
        kwargs_96045 = {}
        
        # Call to LinearNDInterpolator(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'points' (line 102)
        points_96040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'points', False)
        # Getting the type of 'values' (line 102)
        values_96041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'values', False)
        # Processing the call keyword arguments (line 102)
        kwargs_96042 = {}
        # Getting the type of 'interpnd' (line 102)
        interpnd_96038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 102)
        LinearNDInterpolator_96039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 13), interpnd_96038, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 102)
        LinearNDInterpolator_call_result_96043 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), LinearNDInterpolator_96039, *[points_96040, values_96041], **kwargs_96042)
        
        # Calling (args, kwargs) (line 102)
        _call_result_96046 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), LinearNDInterpolator_call_result_96043, *[xi_96044], **kwargs_96045)
        
        # Assigning a type to the variable 'zi' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'zi', _call_result_96046)
        
        # Call to assert_almost_equal(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'zi' (line 104)
        zi_96048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'zi', False)
        
        # Call to ip(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'xx' (line 104)
        xx_96050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'xx', False)
        # Getting the type of 'yy' (line 104)
        yy_96051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'yy', False)
        # Processing the call keyword arguments (line 104)
        kwargs_96052 = {}
        # Getting the type of 'ip' (line 104)
        ip_96049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'ip', False)
        # Calling ip(args, kwargs) (line 104)
        ip_call_result_96053 = invoke(stypy.reporting.localization.Localization(__file__, 104, 32), ip_96049, *[xx_96050, yy_96051], **kwargs_96052)
        
        # Processing the call keyword arguments (line 104)
        kwargs_96054 = {}
        # Getting the type of 'assert_almost_equal' (line 104)
        assert_almost_equal_96047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 104)
        assert_almost_equal_call_result_96055 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), assert_almost_equal_96047, *[zi_96048, ip_call_result_96053], **kwargs_96054)
        
        
        # ################# End of 'test_square(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_96056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square'
        return stypy_return_type_96056


    @norecursion
    def test_smoketest_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoketest_rescale'
        module_type_store = module_type_store.open_function_context('test_smoketest_rescale', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_smoketest_rescale')
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_smoketest_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_smoketest_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoketest_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoketest_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to array(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_96059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_96060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        int_96061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 23), tuple_96060, int_96061)
        # Adding element type (line 108)
        int_96062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 23), tuple_96060, int_96062)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), list_96059, tuple_96060)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_96063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        int_96064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_96063, int_96064)
        # Adding element type (line 108)
        int_96065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 31), tuple_96063, int_96065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), list_96059, tuple_96063)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_96066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        int_96067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 41), tuple_96066, int_96067)
        # Adding element type (line 108)
        int_96068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 41), tuple_96066, int_96068)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), list_96059, tuple_96066)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_96069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        int_96070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 50), tuple_96069, int_96070)
        # Adding element type (line 108)
        int_96071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 50), tuple_96069, int_96071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), list_96059, tuple_96069)
        # Adding element type (line 108)
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_96072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        float_96073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 58), tuple_96072, float_96073)
        # Adding element type (line 108)
        int_96074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 58), tuple_96072, int_96074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 21), list_96059, tuple_96072)
        
        # Processing the call keyword arguments (line 108)
        # Getting the type of 'np' (line 109)
        np_96075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 109)
        double_96076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), np_96075, 'double')
        keyword_96077 = double_96076
        kwargs_96078 = {'dtype': keyword_96077}
        # Getting the type of 'np' (line 108)
        np_96057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 108)
        array_96058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), np_96057, 'array')
        # Calling array(args, kwargs) (line 108)
        array_call_result_96079 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), array_96058, *[list_96059], **kwargs_96078)
        
        # Assigning a type to the variable 'x' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'x', array_call_result_96079)
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to arange(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Obtaining the type of the subscript
        int_96082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        # Getting the type of 'x' (line 110)
        x_96083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 110)
        shape_96084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), x_96083, 'shape')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___96085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 22), shape_96084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_96086 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), getitem___96085, int_96082)
        
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'np' (line 110)
        np_96087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 110)
        double_96088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 40), np_96087, 'double')
        keyword_96089 = double_96088
        kwargs_96090 = {'dtype': keyword_96089}
        # Getting the type of 'np' (line 110)
        np_96080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 110)
        arange_96081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), np_96080, 'arange')
        # Calling arange(args, kwargs) (line 110)
        arange_call_result_96091 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), arange_96081, *[subscript_call_result_96086], **kwargs_96090)
        
        # Assigning a type to the variable 'y' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'y', arange_call_result_96091)
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to (...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_96100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 63), 'x', False)
        # Processing the call keyword arguments (line 112)
        kwargs_96101 = {}
        
        # Call to LinearNDInterpolator(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'x' (line 112)
        x_96094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'x', False)
        # Getting the type of 'y' (line 112)
        y_96095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'y', False)
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'True' (line 112)
        True_96096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 57), 'True', False)
        keyword_96097 = True_96096
        kwargs_96098 = {'rescale': keyword_96097}
        # Getting the type of 'interpnd' (line 112)
        interpnd_96092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 112)
        LinearNDInterpolator_96093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), interpnd_96092, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 112)
        LinearNDInterpolator_call_result_96099 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), LinearNDInterpolator_96093, *[x_96094, y_96095], **kwargs_96098)
        
        # Calling (args, kwargs) (line 112)
        _call_result_96102 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), LinearNDInterpolator_call_result_96099, *[x_96100], **kwargs_96101)
        
        # Assigning a type to the variable 'yi' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'yi', _call_result_96102)
        
        # Call to assert_almost_equal(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'y' (line 113)
        y_96104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'y', False)
        # Getting the type of 'yi' (line 113)
        yi_96105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'yi', False)
        # Processing the call keyword arguments (line 113)
        kwargs_96106 = {}
        # Getting the type of 'assert_almost_equal' (line 113)
        assert_almost_equal_96103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 113)
        assert_almost_equal_call_result_96107 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_almost_equal_96103, *[y_96104, yi_96105], **kwargs_96106)
        
        
        # ################# End of 'test_smoketest_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoketest_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_96108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96108)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoketest_rescale'
        return stypy_return_type_96108


    @norecursion
    def test_square_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_square_rescale'
        module_type_store = module_type_store.open_function_context('test_square_rescale', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_square_rescale')
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_square_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_square_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_square_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_square_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to array(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_96111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_96112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_96113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 28), tuple_96112, int_96113)
        # Adding element type (line 119)
        int_96114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 28), tuple_96112, int_96114)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_96111, tuple_96112)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_96115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_96116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 35), tuple_96115, int_96116)
        # Adding element type (line 119)
        int_96117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 35), tuple_96115, int_96117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_96111, tuple_96115)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_96118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_96119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), tuple_96118, int_96119)
        # Adding element type (line 119)
        int_96120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 44), tuple_96118, int_96120)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_96111, tuple_96118)
        # Adding element type (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_96121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        int_96122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 54), tuple_96121, int_96122)
        # Adding element type (line 119)
        int_96123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 54), tuple_96121, int_96123)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 26), list_96111, tuple_96121)
        
        # Processing the call keyword arguments (line 119)
        # Getting the type of 'np' (line 119)
        np_96124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 68), 'np', False)
        # Obtaining the member 'double' of a type (line 119)
        double_96125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 68), np_96124, 'double')
        keyword_96126 = double_96125
        kwargs_96127 = {'dtype': keyword_96126}
        # Getting the type of 'np' (line 119)
        np_96109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 119)
        array_96110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 17), np_96109, 'array')
        # Calling array(args, kwargs) (line 119)
        array_call_result_96128 = invoke(stypy.reporting.localization.Localization(__file__, 119, 17), array_96110, *[list_96111], **kwargs_96127)
        
        # Assigning a type to the variable 'points' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'points', array_call_result_96128)
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to array(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_96131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        # Adding element type (line 120)
        float_96132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), list_96131, float_96132)
        # Adding element type (line 120)
        float_96133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), list_96131, float_96133)
        # Adding element type (line 120)
        float_96134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), list_96131, float_96134)
        # Adding element type (line 120)
        float_96135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 40), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 26), list_96131, float_96135)
        
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'np' (line 120)
        np_96136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'np', False)
        # Obtaining the member 'double' of a type (line 120)
        double_96137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 51), np_96136, 'double')
        keyword_96138 = double_96137
        kwargs_96139 = {'dtype': keyword_96138}
        # Getting the type of 'np' (line 120)
        np_96129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 120)
        array_96130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), np_96129, 'array')
        # Calling array(args, kwargs) (line 120)
        array_call_result_96140 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), array_96130, *[list_96131], **kwargs_96139)
        
        # Assigning a type to the variable 'values' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'values', array_call_result_96140)
        
        # Assigning a Call to a Tuple (line 122):
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_96141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        slice_96144 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 37), None, None, None)
        # Getting the type of 'None' (line 122)
        None_96145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 62), 'None', False)
        
        # Call to linspace(...): (line 122)
        # Processing the call arguments (line 122)
        int_96148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 49), 'int')
        int_96149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
        int_96150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 56), 'int')
        # Processing the call keyword arguments (line 122)
        kwargs_96151 = {}
        # Getting the type of 'np' (line 122)
        np_96146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 122)
        linspace_96147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 37), np_96146, 'linspace')
        # Calling linspace(args, kwargs) (line 122)
        linspace_call_result_96152 = invoke(stypy.reporting.localization.Localization(__file__, 122, 37), linspace_96147, *[int_96148, int_96149, int_96150], **kwargs_96151)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___96153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 37), linspace_call_result_96152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_96154 = invoke(stypy.reporting.localization.Localization(__file__, 122, 37), getitem___96153, (slice_96144, None_96145))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 123)
        None_96155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 61), 'None', False)
        slice_96156 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 123, 37), None, None, None)
        
        # Call to linspace(...): (line 123)
        # Processing the call arguments (line 123)
        int_96159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        int_96160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'int')
        int_96161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 57), 'int')
        # Processing the call keyword arguments (line 123)
        kwargs_96162 = {}
        # Getting the type of 'np' (line 123)
        np_96157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 123)
        linspace_96158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), np_96157, 'linspace')
        # Calling linspace(args, kwargs) (line 123)
        linspace_call_result_96163 = invoke(stypy.reporting.localization.Localization(__file__, 123, 37), linspace_96158, *[int_96159, int_96160, int_96161], **kwargs_96162)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___96164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), linspace_call_result_96163, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_96165 = invoke(stypy.reporting.localization.Localization(__file__, 123, 37), getitem___96164, (None_96155, slice_96156))
        
        # Processing the call keyword arguments (line 122)
        kwargs_96166 = {}
        # Getting the type of 'np' (line 122)
        np_96142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 122)
        broadcast_arrays_96143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), np_96142, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 122)
        broadcast_arrays_call_result_96167 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), broadcast_arrays_96143, *[subscript_call_result_96154, subscript_call_result_96165], **kwargs_96166)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___96168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), broadcast_arrays_call_result_96167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_96169 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), getitem___96168, int_96141)
        
        # Assigning a type to the variable 'tuple_var_assignment_95576' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_95576', subscript_call_result_96169)
        
        # Assigning a Subscript to a Name (line 122):
        
        # Obtaining the type of the subscript
        int_96170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining the type of the subscript
        slice_96173 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 37), None, None, None)
        # Getting the type of 'None' (line 122)
        None_96174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 62), 'None', False)
        
        # Call to linspace(...): (line 122)
        # Processing the call arguments (line 122)
        int_96177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 49), 'int')
        int_96178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
        int_96179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 56), 'int')
        # Processing the call keyword arguments (line 122)
        kwargs_96180 = {}
        # Getting the type of 'np' (line 122)
        np_96175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 122)
        linspace_96176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 37), np_96175, 'linspace')
        # Calling linspace(args, kwargs) (line 122)
        linspace_call_result_96181 = invoke(stypy.reporting.localization.Localization(__file__, 122, 37), linspace_96176, *[int_96177, int_96178, int_96179], **kwargs_96180)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___96182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 37), linspace_call_result_96181, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_96183 = invoke(stypy.reporting.localization.Localization(__file__, 122, 37), getitem___96182, (slice_96173, None_96174))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 123)
        None_96184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 61), 'None', False)
        slice_96185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 123, 37), None, None, None)
        
        # Call to linspace(...): (line 123)
        # Processing the call arguments (line 123)
        int_96188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        int_96189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'int')
        int_96190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 57), 'int')
        # Processing the call keyword arguments (line 123)
        kwargs_96191 = {}
        # Getting the type of 'np' (line 123)
        np_96186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'np', False)
        # Obtaining the member 'linspace' of a type (line 123)
        linspace_96187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), np_96186, 'linspace')
        # Calling linspace(args, kwargs) (line 123)
        linspace_call_result_96192 = invoke(stypy.reporting.localization.Localization(__file__, 123, 37), linspace_96187, *[int_96188, int_96189, int_96190], **kwargs_96191)
        
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___96193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), linspace_call_result_96192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_96194 = invoke(stypy.reporting.localization.Localization(__file__, 123, 37), getitem___96193, (None_96184, slice_96185))
        
        # Processing the call keyword arguments (line 122)
        kwargs_96195 = {}
        # Getting the type of 'np' (line 122)
        np_96171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 122)
        broadcast_arrays_96172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), np_96171, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 122)
        broadcast_arrays_call_result_96196 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), broadcast_arrays_96172, *[subscript_call_result_96183, subscript_call_result_96194], **kwargs_96195)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___96197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), broadcast_arrays_call_result_96196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_96198 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), getitem___96197, int_96170)
        
        # Assigning a type to the variable 'tuple_var_assignment_95577' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_95577', subscript_call_result_96198)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_95576' (line 122)
        tuple_var_assignment_95576_96199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_95576')
        # Assigning a type to the variable 'xx' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'xx', tuple_var_assignment_95576_96199)
        
        # Assigning a Name to a Name (line 122):
        # Getting the type of 'tuple_var_assignment_95577' (line 122)
        tuple_var_assignment_95577_96200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'tuple_var_assignment_95577')
        # Assigning a type to the variable 'yy' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'yy', tuple_var_assignment_95577_96200)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to ravel(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_96203 = {}
        # Getting the type of 'xx' (line 124)
        xx_96201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'xx', False)
        # Obtaining the member 'ravel' of a type (line 124)
        ravel_96202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), xx_96201, 'ravel')
        # Calling ravel(args, kwargs) (line 124)
        ravel_call_result_96204 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), ravel_96202, *[], **kwargs_96203)
        
        # Assigning a type to the variable 'xx' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'xx', ravel_call_result_96204)
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to ravel(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_96207 = {}
        # Getting the type of 'yy' (line 125)
        yy_96205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'yy', False)
        # Obtaining the member 'ravel' of a type (line 125)
        ravel_96206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), yy_96205, 'ravel')
        # Calling ravel(args, kwargs) (line 125)
        ravel_call_result_96208 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), ravel_96206, *[], **kwargs_96207)
        
        # Assigning a type to the variable 'yy' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'yy', ravel_call_result_96208)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to copy(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_96218 = {}
        
        # Call to array(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_96211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        # Getting the type of 'xx' (line 126)
        xx_96212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'xx', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 22), list_96211, xx_96212)
        # Adding element type (line 126)
        # Getting the type of 'yy' (line 126)
        yy_96213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'yy', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 22), list_96211, yy_96213)
        
        # Processing the call keyword arguments (line 126)
        kwargs_96214 = {}
        # Getting the type of 'np' (line 126)
        np_96209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 126)
        array_96210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), np_96209, 'array')
        # Calling array(args, kwargs) (line 126)
        array_call_result_96215 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), array_96210, *[list_96211], **kwargs_96214)
        
        # Obtaining the member 'T' of a type (line 126)
        T_96216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), array_call_result_96215, 'T')
        # Obtaining the member 'copy' of a type (line 126)
        copy_96217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), T_96216, 'copy')
        # Calling copy(args, kwargs) (line 126)
        copy_call_result_96219 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), copy_96217, *[], **kwargs_96218)
        
        # Assigning a type to the variable 'xi' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'xi', copy_call_result_96219)
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to (...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'xi' (line 127)
        xi_96226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 59), 'xi', False)
        # Processing the call keyword arguments (line 127)
        kwargs_96227 = {}
        
        # Call to LinearNDInterpolator(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'points' (line 127)
        points_96222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 43), 'points', False)
        # Getting the type of 'values' (line 127)
        values_96223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 51), 'values', False)
        # Processing the call keyword arguments (line 127)
        kwargs_96224 = {}
        # Getting the type of 'interpnd' (line 127)
        interpnd_96220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 127)
        LinearNDInterpolator_96221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 13), interpnd_96220, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 127)
        LinearNDInterpolator_call_result_96225 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), LinearNDInterpolator_96221, *[points_96222, values_96223], **kwargs_96224)
        
        # Calling (args, kwargs) (line 127)
        _call_result_96228 = invoke(stypy.reporting.localization.Localization(__file__, 127, 13), LinearNDInterpolator_call_result_96225, *[xi_96226], **kwargs_96227)
        
        # Assigning a type to the variable 'zi' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'zi', _call_result_96228)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to (...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'xi' (line 129)
        xi_96237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'xi', False)
        # Processing the call keyword arguments (line 128)
        kwargs_96238 = {}
        
        # Call to LinearNDInterpolator(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'points' (line 128)
        points_96231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'points', False)
        # Getting the type of 'values' (line 128)
        values_96232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'values', False)
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'True' (line 129)
        True_96233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'True', False)
        keyword_96234 = True_96233
        kwargs_96235 = {'rescale': keyword_96234}
        # Getting the type of 'interpnd' (line 128)
        interpnd_96229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 128)
        LinearNDInterpolator_96230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 22), interpnd_96229, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 128)
        LinearNDInterpolator_call_result_96236 = invoke(stypy.reporting.localization.Localization(__file__, 128, 22), LinearNDInterpolator_96230, *[points_96231, values_96232], **kwargs_96235)
        
        # Calling (args, kwargs) (line 128)
        _call_result_96239 = invoke(stypy.reporting.localization.Localization(__file__, 128, 22), LinearNDInterpolator_call_result_96236, *[xi_96237], **kwargs_96238)
        
        # Assigning a type to the variable 'zi_rescaled' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'zi_rescaled', _call_result_96239)
        
        # Call to assert_almost_equal(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'zi' (line 131)
        zi_96241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'zi', False)
        # Getting the type of 'zi_rescaled' (line 131)
        zi_rescaled_96242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 32), 'zi_rescaled', False)
        # Processing the call keyword arguments (line 131)
        kwargs_96243 = {}
        # Getting the type of 'assert_almost_equal' (line 131)
        assert_almost_equal_96240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 131)
        assert_almost_equal_call_result_96244 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assert_almost_equal_96240, *[zi_96241, zi_rescaled_96242], **kwargs_96243)
        
        
        # ################# End of 'test_square_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_square_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_96245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_square_rescale'
        return stypy_return_type_96245


    @norecursion
    def test_tripoints_input_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tripoints_input_rescale'
        module_type_store = module_type_store.open_function_context('test_tripoints_input_rescale', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_tripoints_input_rescale')
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_tripoints_input_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_tripoints_input_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tripoints_input_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tripoints_input_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to array(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_96248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_96249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        int_96250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), tuple_96249, int_96250)
        # Adding element type (line 135)
        int_96251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 23), tuple_96249, int_96251)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_96248, tuple_96249)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_96252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        int_96253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 30), tuple_96252, int_96253)
        # Adding element type (line 135)
        int_96254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 30), tuple_96252, int_96254)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_96248, tuple_96252)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_96255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        int_96256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 39), tuple_96255, int_96256)
        # Adding element type (line 135)
        int_96257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 39), tuple_96255, int_96257)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_96248, tuple_96255)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_96258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        int_96259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 47), tuple_96258, int_96259)
        # Adding element type (line 135)
        int_96260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 47), tuple_96258, int_96260)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_96248, tuple_96258)
        # Adding element type (line 135)
        
        # Obtaining an instance of the builtin type 'tuple' (line 135)
        tuple_96261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 135)
        # Adding element type (line 135)
        float_96262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 55), tuple_96261, float_96262)
        # Adding element type (line 135)
        int_96263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 55), tuple_96261, int_96263)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_96248, tuple_96261)
        
        # Processing the call keyword arguments (line 135)
        # Getting the type of 'np' (line 136)
        np_96264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 136)
        double_96265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 27), np_96264, 'double')
        keyword_96266 = double_96265
        kwargs_96267 = {'dtype': keyword_96266}
        # Getting the type of 'np' (line 135)
        np_96246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 135)
        array_96247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), np_96246, 'array')
        # Calling array(args, kwargs) (line 135)
        array_call_result_96268 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), array_96247, *[list_96248], **kwargs_96267)
        
        # Assigning a type to the variable 'x' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'x', array_call_result_96268)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to arange(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        int_96271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'int')
        # Getting the type of 'x' (line 137)
        x_96272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 137)
        shape_96273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 22), x_96272, 'shape')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___96274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 22), shape_96273, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_96275 = invoke(stypy.reporting.localization.Localization(__file__, 137, 22), getitem___96274, int_96271)
        
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'np' (line 137)
        np_96276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 137)
        double_96277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 40), np_96276, 'double')
        keyword_96278 = double_96277
        kwargs_96279 = {'dtype': keyword_96278}
        # Getting the type of 'np' (line 137)
        np_96269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 137)
        arange_96270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 12), np_96269, 'arange')
        # Calling arange(args, kwargs) (line 137)
        arange_call_result_96280 = invoke(stypy.reporting.localization.Localization(__file__, 137, 12), arange_96270, *[subscript_call_result_96275], **kwargs_96279)
        
        # Assigning a type to the variable 'y' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'y', arange_call_result_96280)
        
        # Assigning a BinOp to a Name (line 138):
        
        # Assigning a BinOp to a Name (line 138):
        # Getting the type of 'y' (line 138)
        y_96281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'y')
        complex_96282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 16), 'complex')
        # Getting the type of 'y' (line 138)
        y_96283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'y')
        # Applying the binary operator '*' (line 138)
        result_mul_96284 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 16), '*', complex_96282, y_96283)
        
        # Applying the binary operator '-' (line 138)
        result_sub_96285 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 12), '-', y_96281, result_mul_96284)
        
        # Assigning a type to the variable 'y' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'y', result_sub_96285)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to Delaunay(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'x' (line 140)
        x_96288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'x', False)
        # Processing the call keyword arguments (line 140)
        kwargs_96289 = {}
        # Getting the type of 'qhull' (line 140)
        qhull_96286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 140)
        Delaunay_96287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 14), qhull_96286, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 140)
        Delaunay_call_result_96290 = invoke(stypy.reporting.localization.Localization(__file__, 140, 14), Delaunay_96287, *[x_96288], **kwargs_96289)
        
        # Assigning a type to the variable 'tri' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tri', Delaunay_call_result_96290)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to (...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'x' (line 141)
        x_96298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 58), 'x', False)
        # Processing the call keyword arguments (line 141)
        kwargs_96299 = {}
        
        # Call to LinearNDInterpolator(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'tri' (line 141)
        tri_96293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 43), 'tri', False)
        # Obtaining the member 'points' of a type (line 141)
        points_96294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 43), tri_96293, 'points')
        # Getting the type of 'y' (line 141)
        y_96295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 55), 'y', False)
        # Processing the call keyword arguments (line 141)
        kwargs_96296 = {}
        # Getting the type of 'interpnd' (line 141)
        interpnd_96291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 141)
        LinearNDInterpolator_96292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 13), interpnd_96291, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 141)
        LinearNDInterpolator_call_result_96297 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), LinearNDInterpolator_96292, *[points_96294, y_96295], **kwargs_96296)
        
        # Calling (args, kwargs) (line 141)
        _call_result_96300 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), LinearNDInterpolator_call_result_96297, *[x_96298], **kwargs_96299)
        
        # Assigning a type to the variable 'yi' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'yi', _call_result_96300)
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to (...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 143)
        x_96310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_96311 = {}
        
        # Call to LinearNDInterpolator(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'tri' (line 142)
        tri_96303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 51), 'tri', False)
        # Obtaining the member 'points' of a type (line 142)
        points_96304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 51), tri_96303, 'points')
        # Getting the type of 'y' (line 142)
        y_96305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 63), 'y', False)
        # Processing the call keyword arguments (line 142)
        # Getting the type of 'True' (line 143)
        True_96306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'True', False)
        keyword_96307 = True_96306
        kwargs_96308 = {'rescale': keyword_96307}
        # Getting the type of 'interpnd' (line 142)
        interpnd_96301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 142)
        LinearNDInterpolator_96302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 21), interpnd_96301, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 142)
        LinearNDInterpolator_call_result_96309 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), LinearNDInterpolator_96302, *[points_96304, y_96305], **kwargs_96308)
        
        # Calling (args, kwargs) (line 142)
        _call_result_96312 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), LinearNDInterpolator_call_result_96309, *[x_96310], **kwargs_96311)
        
        # Assigning a type to the variable 'yi_rescale' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'yi_rescale', _call_result_96312)
        
        # Call to assert_almost_equal(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'yi' (line 144)
        yi_96314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'yi', False)
        # Getting the type of 'yi_rescale' (line 144)
        yi_rescale_96315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'yi_rescale', False)
        # Processing the call keyword arguments (line 144)
        kwargs_96316 = {}
        # Getting the type of 'assert_almost_equal' (line 144)
        assert_almost_equal_96313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 144)
        assert_almost_equal_call_result_96317 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assert_almost_equal_96313, *[yi_96314, yi_rescale_96315], **kwargs_96316)
        
        
        # ################# End of 'test_tripoints_input_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tripoints_input_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_96318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96318)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tripoints_input_rescale'
        return stypy_return_type_96318


    @norecursion
    def test_tri_input_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tri_input_rescale'
        module_type_store = module_type_store.open_function_context('test_tri_input_rescale', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_tri_input_rescale')
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_tri_input_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_tri_input_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tri_input_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tri_input_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to array(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Obtaining an instance of the builtin type 'list' (line 148)
        list_96321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 148)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_96322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        int_96323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), tuple_96322, int_96323)
        # Adding element type (line 148)
        int_96324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 23), tuple_96322, int_96324)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 21), list_96321, tuple_96322)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_96325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        int_96326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 30), tuple_96325, int_96326)
        # Adding element type (line 148)
        int_96327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 30), tuple_96325, int_96327)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 21), list_96321, tuple_96325)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_96328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        int_96329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 39), tuple_96328, int_96329)
        # Adding element type (line 148)
        int_96330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 39), tuple_96328, int_96330)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 21), list_96321, tuple_96328)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_96331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        int_96332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 47), tuple_96331, int_96332)
        # Adding element type (line 148)
        int_96333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 47), tuple_96331, int_96333)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 21), list_96321, tuple_96331)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_96334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        float_96335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 55), tuple_96334, float_96335)
        # Adding element type (line 148)
        int_96336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 55), tuple_96334, int_96336)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 21), list_96321, tuple_96334)
        
        # Processing the call keyword arguments (line 148)
        # Getting the type of 'np' (line 149)
        np_96337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 149)
        double_96338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 27), np_96337, 'double')
        keyword_96339 = double_96338
        kwargs_96340 = {'dtype': keyword_96339}
        # Getting the type of 'np' (line 148)
        np_96319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 148)
        array_96320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), np_96319, 'array')
        # Calling array(args, kwargs) (line 148)
        array_call_result_96341 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), array_96320, *[list_96321], **kwargs_96340)
        
        # Assigning a type to the variable 'x' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'x', array_call_result_96341)
        
        # Assigning a Call to a Name (line 150):
        
        # Assigning a Call to a Name (line 150):
        
        # Call to arange(...): (line 150)
        # Processing the call arguments (line 150)
        
        # Obtaining the type of the subscript
        int_96344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'int')
        # Getting the type of 'x' (line 150)
        x_96345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 150)
        shape_96346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), x_96345, 'shape')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___96347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), shape_96346, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_96348 = invoke(stypy.reporting.localization.Localization(__file__, 150, 22), getitem___96347, int_96344)
        
        # Processing the call keyword arguments (line 150)
        # Getting the type of 'np' (line 150)
        np_96349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 150)
        double_96350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 40), np_96349, 'double')
        keyword_96351 = double_96350
        kwargs_96352 = {'dtype': keyword_96351}
        # Getting the type of 'np' (line 150)
        np_96342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 150)
        arange_96343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), np_96342, 'arange')
        # Calling arange(args, kwargs) (line 150)
        arange_call_result_96353 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), arange_96343, *[subscript_call_result_96348], **kwargs_96352)
        
        # Assigning a type to the variable 'y' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'y', arange_call_result_96353)
        
        # Assigning a BinOp to a Name (line 151):
        
        # Assigning a BinOp to a Name (line 151):
        # Getting the type of 'y' (line 151)
        y_96354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'y')
        complex_96355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 16), 'complex')
        # Getting the type of 'y' (line 151)
        y_96356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 19), 'y')
        # Applying the binary operator '*' (line 151)
        result_mul_96357 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 16), '*', complex_96355, y_96356)
        
        # Applying the binary operator '-' (line 151)
        result_sub_96358 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 12), '-', y_96354, result_mul_96357)
        
        # Assigning a type to the variable 'y' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'y', result_sub_96358)
        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to Delaunay(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'x' (line 153)
        x_96361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'x', False)
        # Processing the call keyword arguments (line 153)
        kwargs_96362 = {}
        # Getting the type of 'qhull' (line 153)
        qhull_96359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 153)
        Delaunay_96360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 14), qhull_96359, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 153)
        Delaunay_call_result_96363 = invoke(stypy.reporting.localization.Localization(__file__, 153, 14), Delaunay_96360, *[x_96361], **kwargs_96362)
        
        # Assigning a type to the variable 'tri' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'tri', Delaunay_call_result_96363)
        
        
        # SSA begins for try-except statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to (...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_96372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 64), 'x', False)
        # Processing the call keyword arguments (line 155)
        kwargs_96373 = {}
        
        # Call to LinearNDInterpolator(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'tri' (line 155)
        tri_96366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'tri', False)
        # Getting the type of 'y' (line 155)
        y_96367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 47), 'y', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'True' (line 155)
        True_96368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 58), 'True', False)
        keyword_96369 = True_96368
        kwargs_96370 = {'rescale': keyword_96369}
        # Getting the type of 'interpnd' (line 155)
        interpnd_96364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 155)
        LinearNDInterpolator_96365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), interpnd_96364, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 155)
        LinearNDInterpolator_call_result_96371 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), LinearNDInterpolator_96365, *[tri_96366, y_96367], **kwargs_96370)
        
        # Calling (args, kwargs) (line 155)
        _call_result_96374 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), LinearNDInterpolator_call_result_96371, *[x_96372], **kwargs_96373)
        
        # SSA branch for the except part of a try statement (line 154)
        # SSA branch for the except 'ValueError' branch of a try statement (line 154)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 156)
        ValueError_96375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'ValueError')
        # Assigning a type to the variable 'e' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'e', ValueError_96375)
        
        
        
        # Call to str(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'e' (line 157)
        e_96377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'e', False)
        # Processing the call keyword arguments (line 157)
        kwargs_96378 = {}
        # Getting the type of 'str' (line 157)
        str_96376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'str', False)
        # Calling str(args, kwargs) (line 157)
        str_call_result_96379 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), str_96376, *[e_96377], **kwargs_96378)
        
        str_96380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'str', 'Rescaling is not supported when passing a Delaunay triangulation as ``points``.')
        # Applying the binary operator '!=' (line 157)
        result_ne_96381 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), '!=', str_call_result_96379, str_96380)
        
        # Testing the type of an if condition (line 157)
        if_condition_96382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_ne_96381)
        # Assigning a type to the variable 'if_condition_96382' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_96382', if_condition_96382)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except '<any exception>' branch of a try statement (line 154)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tri_input_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tri_input_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_96383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96383)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tri_input_rescale'
        return stypy_return_type_96383


    @norecursion
    def test_pickle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pickle'
        module_type_store = module_type_store.open_function_context('test_pickle', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_localization', localization)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_function_name', 'TestLinearNDInterpolation.test_pickle')
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_param_names_list', [])
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLinearNDInterpolation.test_pickle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.test_pickle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pickle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pickle(...)' code ##################

        
        # Call to seed(...): (line 165)
        # Processing the call arguments (line 165)
        int_96387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 23), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_96388 = {}
        # Getting the type of 'np' (line 165)
        np_96384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 165)
        random_96385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), np_96384, 'random')
        # Obtaining the member 'seed' of a type (line 165)
        seed_96386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), random_96385, 'seed')
        # Calling seed(args, kwargs) (line 165)
        seed_call_result_96389 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), seed_96386, *[int_96387], **kwargs_96388)
        
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to rand(...): (line 166)
        # Processing the call arguments (line 166)
        int_96393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 27), 'int')
        int_96394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 31), 'int')
        # Processing the call keyword arguments (line 166)
        kwargs_96395 = {}
        # Getting the type of 'np' (line 166)
        np_96390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 166)
        random_96391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), np_96390, 'random')
        # Obtaining the member 'rand' of a type (line 166)
        rand_96392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), random_96391, 'rand')
        # Calling rand(args, kwargs) (line 166)
        rand_call_result_96396 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), rand_96392, *[int_96393, int_96394], **kwargs_96395)
        
        # Assigning a type to the variable 'x' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'x', rand_call_result_96396)
        
        # Assigning a BinOp to a Name (line 167):
        
        # Assigning a BinOp to a Name (line 167):
        
        # Call to rand(...): (line 167)
        # Processing the call arguments (line 167)
        int_96400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 27), 'int')
        # Processing the call keyword arguments (line 167)
        kwargs_96401 = {}
        # Getting the type of 'np' (line 167)
        np_96397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 167)
        random_96398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), np_96397, 'random')
        # Obtaining the member 'rand' of a type (line 167)
        rand_96399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), random_96398, 'rand')
        # Calling rand(args, kwargs) (line 167)
        rand_call_result_96402 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), rand_96399, *[int_96400], **kwargs_96401)
        
        complex_96403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 33), 'complex')
        
        # Call to rand(...): (line 167)
        # Processing the call arguments (line 167)
        int_96407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 51), 'int')
        # Processing the call keyword arguments (line 167)
        kwargs_96408 = {}
        # Getting the type of 'np' (line 167)
        np_96404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 36), 'np', False)
        # Obtaining the member 'random' of a type (line 167)
        random_96405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 36), np_96404, 'random')
        # Obtaining the member 'rand' of a type (line 167)
        rand_96406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 36), random_96405, 'rand')
        # Calling rand(args, kwargs) (line 167)
        rand_call_result_96409 = invoke(stypy.reporting.localization.Localization(__file__, 167, 36), rand_96406, *[int_96407], **kwargs_96408)
        
        # Applying the binary operator '*' (line 167)
        result_mul_96410 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 33), '*', complex_96403, rand_call_result_96409)
        
        # Applying the binary operator '+' (line 167)
        result_add_96411 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 12), '+', rand_call_result_96402, result_mul_96410)
        
        # Assigning a type to the variable 'y' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'y', result_add_96411)
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to LinearNDInterpolator(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'x' (line 169)
        x_96414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'x', False)
        # Getting the type of 'y' (line 169)
        y_96415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 46), 'y', False)
        # Processing the call keyword arguments (line 169)
        kwargs_96416 = {}
        # Getting the type of 'interpnd' (line 169)
        interpnd_96412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'interpnd', False)
        # Obtaining the member 'LinearNDInterpolator' of a type (line 169)
        LinearNDInterpolator_96413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 13), interpnd_96412, 'LinearNDInterpolator')
        # Calling LinearNDInterpolator(args, kwargs) (line 169)
        LinearNDInterpolator_call_result_96417 = invoke(stypy.reporting.localization.Localization(__file__, 169, 13), LinearNDInterpolator_96413, *[x_96414, y_96415], **kwargs_96416)
        
        # Assigning a type to the variable 'ip' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'ip', LinearNDInterpolator_call_result_96417)
        
        # Assigning a Call to a Name (line 170):
        
        # Assigning a Call to a Name (line 170):
        
        # Call to loads(...): (line 170)
        # Processing the call arguments (line 170)
        
        # Call to dumps(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'ip' (line 170)
        ip_96422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 40), 'ip', False)
        # Processing the call keyword arguments (line 170)
        kwargs_96423 = {}
        # Getting the type of 'pickle' (line 170)
        pickle_96420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 27), 'pickle', False)
        # Obtaining the member 'dumps' of a type (line 170)
        dumps_96421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 27), pickle_96420, 'dumps')
        # Calling dumps(args, kwargs) (line 170)
        dumps_call_result_96424 = invoke(stypy.reporting.localization.Localization(__file__, 170, 27), dumps_96421, *[ip_96422], **kwargs_96423)
        
        # Processing the call keyword arguments (line 170)
        kwargs_96425 = {}
        # Getting the type of 'pickle' (line 170)
        pickle_96418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'pickle', False)
        # Obtaining the member 'loads' of a type (line 170)
        loads_96419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 14), pickle_96418, 'loads')
        # Calling loads(args, kwargs) (line 170)
        loads_call_result_96426 = invoke(stypy.reporting.localization.Localization(__file__, 170, 14), loads_96419, *[dumps_call_result_96424], **kwargs_96425)
        
        # Assigning a type to the variable 'ip2' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'ip2', loads_call_result_96426)
        
        # Call to assert_almost_equal(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to ip(...): (line 172)
        # Processing the call arguments (line 172)
        float_96429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'float')
        float_96430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 36), 'float')
        # Processing the call keyword arguments (line 172)
        kwargs_96431 = {}
        # Getting the type of 'ip' (line 172)
        ip_96428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'ip', False)
        # Calling ip(args, kwargs) (line 172)
        ip_call_result_96432 = invoke(stypy.reporting.localization.Localization(__file__, 172, 28), ip_96428, *[float_96429, float_96430], **kwargs_96431)
        
        
        # Call to ip2(...): (line 172)
        # Processing the call arguments (line 172)
        float_96434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 46), 'float')
        float_96435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 51), 'float')
        # Processing the call keyword arguments (line 172)
        kwargs_96436 = {}
        # Getting the type of 'ip2' (line 172)
        ip2_96433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'ip2', False)
        # Calling ip2(args, kwargs) (line 172)
        ip2_call_result_96437 = invoke(stypy.reporting.localization.Localization(__file__, 172, 42), ip2_96433, *[float_96434, float_96435], **kwargs_96436)
        
        # Processing the call keyword arguments (line 172)
        kwargs_96438 = {}
        # Getting the type of 'assert_almost_equal' (line 172)
        assert_almost_equal_96427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 172)
        assert_almost_equal_call_result_96439 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert_almost_equal_96427, *[ip_call_result_96432, ip2_call_result_96437], **kwargs_96438)
        
        
        # ################# End of 'test_pickle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pickle' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_96440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96440)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pickle'
        return stypy_return_type_96440


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLinearNDInterpolation.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLinearNDInterpolation' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'TestLinearNDInterpolation', TestLinearNDInterpolation)
# Declaration of the 'TestEstimateGradients2DGlobal' class

class TestEstimateGradients2DGlobal(object, ):

    @norecursion
    def test_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_smoketest'
        module_type_store = module_type_store.open_function_context('test_smoketest', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_function_name', 'TestEstimateGradients2DGlobal.test_smoketest')
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEstimateGradients2DGlobal.test_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEstimateGradients2DGlobal.test_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_smoketest(...)' code ##################

        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to array(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_96443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_96444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        int_96445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 23), tuple_96444, int_96445)
        # Adding element type (line 177)
        int_96446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 23), tuple_96444, int_96446)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96444)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_96447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        int_96448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), tuple_96447, int_96448)
        # Adding element type (line 177)
        int_96449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), tuple_96447, int_96449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96447)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_96450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        int_96451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), tuple_96450, int_96451)
        # Adding element type (line 178)
        int_96452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 23), tuple_96450, int_96452)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96450)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_96453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        int_96454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), tuple_96453, int_96454)
        # Adding element type (line 178)
        int_96455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 31), tuple_96453, int_96455)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96453)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_96456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        float_96457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 39), tuple_96456, float_96457)
        # Adding element type (line 178)
        float_96458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 39), tuple_96456, float_96458)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96456)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_96459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        float_96460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 53), tuple_96459, float_96460)
        # Adding element type (line 178)
        float_96461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 53), tuple_96459, float_96461)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_96443, tuple_96459)
        
        # Processing the call keyword arguments (line 177)
        # Getting the type of 'float' (line 178)
        float_96462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 71), 'float', False)
        keyword_96463 = float_96462
        kwargs_96464 = {'dtype': keyword_96463}
        # Getting the type of 'np' (line 177)
        np_96441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 177)
        array_96442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), np_96441, 'array')
        # Calling array(args, kwargs) (line 177)
        array_call_result_96465 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), array_96442, *[list_96443], **kwargs_96464)
        
        # Assigning a type to the variable 'x' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'x', array_call_result_96465)
        
        # Assigning a Call to a Name (line 179):
        
        # Assigning a Call to a Name (line 179):
        
        # Call to Delaunay(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'x' (line 179)
        x_96468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'x', False)
        # Processing the call keyword arguments (line 179)
        kwargs_96469 = {}
        # Getting the type of 'qhull' (line 179)
        qhull_96466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 179)
        Delaunay_96467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 14), qhull_96466, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 179)
        Delaunay_call_result_96470 = invoke(stypy.reporting.localization.Localization(__file__, 179, 14), Delaunay_96467, *[x_96468], **kwargs_96469)
        
        # Assigning a type to the variable 'tri' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'tri', Delaunay_call_result_96470)
        
        # Assigning a List to a Name (line 183):
        
        # Assigning a List to a Name (line 183):
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_96471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_96472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)

        @norecursion
        def _stypy_temp_lambda_66(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_66'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_66', 184, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_66.stypy_localization = localization
            _stypy_temp_lambda_66.stypy_type_of_self = None
            _stypy_temp_lambda_66.stypy_type_store = module_type_store
            _stypy_temp_lambda_66.stypy_function_name = '_stypy_temp_lambda_66'
            _stypy_temp_lambda_66.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_66.stypy_varargs_param_name = None
            _stypy_temp_lambda_66.stypy_kwargs_param_name = None
            _stypy_temp_lambda_66.stypy_call_defaults = defaults
            _stypy_temp_lambda_66.stypy_call_varargs = varargs
            _stypy_temp_lambda_66.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_66', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_66', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 26), 'int')
            # Getting the type of 'x' (line 184)
            x_96474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'x')
            # Applying the binary operator '*' (line 184)
            result_mul_96475 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 26), '*', int_96473, x_96474)
            
            int_96476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 32), 'int')
            # Applying the binary operator '+' (line 184)
            result_add_96477 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 26), '+', result_mul_96475, int_96476)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'stypy_return_type', result_add_96477)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_66' in the type store
            # Getting the type of 'stypy_return_type' (line 184)
            stypy_return_type_96478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96478)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_66'
            return stypy_return_type_96478

        # Assigning a type to the variable '_stypy_temp_lambda_66' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), '_stypy_temp_lambda_66', _stypy_temp_lambda_66)
        # Getting the type of '_stypy_temp_lambda_66' (line 184)
        _stypy_temp_lambda_66_96479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), '_stypy_temp_lambda_66')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 13), tuple_96472, _stypy_temp_lambda_66_96479)
        # Adding element type (line 184)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_96480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        int_96481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 36), tuple_96480, int_96481)
        # Adding element type (line 184)
        int_96482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 36), tuple_96480, int_96482)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 13), tuple_96472, tuple_96480)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_96471, tuple_96472)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_96483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)

        @norecursion
        def _stypy_temp_lambda_67(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_67'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_67', 185, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_67.stypy_localization = localization
            _stypy_temp_lambda_67.stypy_type_of_self = None
            _stypy_temp_lambda_67.stypy_type_store = module_type_store
            _stypy_temp_lambda_67.stypy_function_name = '_stypy_temp_lambda_67'
            _stypy_temp_lambda_67.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_67.stypy_varargs_param_name = None
            _stypy_temp_lambda_67.stypy_kwargs_param_name = None
            _stypy_temp_lambda_67.stypy_call_defaults = defaults
            _stypy_temp_lambda_67.stypy_call_varargs = varargs
            _stypy_temp_lambda_67.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_67', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_67', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 26), 'int')
            # Getting the type of 'x' (line 185)
            x_96485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'x')
            # Applying the binary operator '+' (line 185)
            result_add_96486 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 26), '+', int_96484, x_96485)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'stypy_return_type', result_add_96486)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_67' in the type store
            # Getting the type of 'stypy_return_type' (line 185)
            stypy_return_type_96487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96487)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_67'
            return stypy_return_type_96487

        # Assigning a type to the variable '_stypy_temp_lambda_67' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), '_stypy_temp_lambda_67', _stypy_temp_lambda_67)
        # Getting the type of '_stypy_temp_lambda_67' (line 185)
        _stypy_temp_lambda_67_96488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 13), '_stypy_temp_lambda_67')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 13), tuple_96483, _stypy_temp_lambda_67_96488)
        # Adding element type (line 185)
        
        # Obtaining an instance of the builtin type 'tuple' (line 185)
        tuple_96489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 185)
        # Adding element type (line 185)
        int_96490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 34), tuple_96489, int_96490)
        # Adding element type (line 185)
        int_96491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 34), tuple_96489, int_96491)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 13), tuple_96483, tuple_96489)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_96471, tuple_96483)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_96492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)

        @norecursion
        def _stypy_temp_lambda_68(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_68'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_68', 186, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_68.stypy_localization = localization
            _stypy_temp_lambda_68.stypy_type_of_self = None
            _stypy_temp_lambda_68.stypy_type_store = module_type_store
            _stypy_temp_lambda_68.stypy_function_name = '_stypy_temp_lambda_68'
            _stypy_temp_lambda_68.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_68.stypy_varargs_param_name = None
            _stypy_temp_lambda_68.stypy_kwargs_param_name = None
            _stypy_temp_lambda_68.stypy_call_defaults = defaults
            _stypy_temp_lambda_68.stypy_call_varargs = varargs
            _stypy_temp_lambda_68.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_68', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_68', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 26), 'int')
            # Getting the type of 'y' (line 186)
            y_96494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'y')
            # Applying the binary operator '+' (line 186)
            result_add_96495 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 26), '+', int_96493, y_96494)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'stypy_return_type', result_add_96495)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_68' in the type store
            # Getting the type of 'stypy_return_type' (line 186)
            stypy_return_type_96496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96496)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_68'
            return stypy_return_type_96496

        # Assigning a type to the variable '_stypy_temp_lambda_68' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), '_stypy_temp_lambda_68', _stypy_temp_lambda_68)
        # Getting the type of '_stypy_temp_lambda_68' (line 186)
        _stypy_temp_lambda_68_96497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), '_stypy_temp_lambda_68')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 13), tuple_96492, _stypy_temp_lambda_68_96497)
        # Adding element type (line 186)
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_96498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        int_96499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 35), tuple_96498, int_96499)
        # Adding element type (line 186)
        int_96500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 35), tuple_96498, int_96500)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 13), tuple_96492, tuple_96498)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_96471, tuple_96492)
        # Adding element type (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_96501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)

        @norecursion
        def _stypy_temp_lambda_69(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_69'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_69', 187, 13, True)
            # Passed parameters checking function
            _stypy_temp_lambda_69.stypy_localization = localization
            _stypy_temp_lambda_69.stypy_type_of_self = None
            _stypy_temp_lambda_69.stypy_type_store = module_type_store
            _stypy_temp_lambda_69.stypy_function_name = '_stypy_temp_lambda_69'
            _stypy_temp_lambda_69.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_69.stypy_varargs_param_name = None
            _stypy_temp_lambda_69.stypy_kwargs_param_name = None
            _stypy_temp_lambda_69.stypy_call_defaults = defaults
            _stypy_temp_lambda_69.stypy_call_varargs = varargs
            _stypy_temp_lambda_69.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_69', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_69', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'int')
            int_96503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 30), 'int')
            # Getting the type of 'x' (line 187)
            x_96504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 32), 'x')
            # Applying the binary operator '*' (line 187)
            result_mul_96505 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 30), '*', int_96503, x_96504)
            
            # Applying the binary operator '+' (line 187)
            result_add_96506 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 26), '+', int_96502, result_mul_96505)
            
            float_96507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 36), 'float')
            # Getting the type of 'y' (line 187)
            y_96508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 42), 'y')
            # Applying the binary operator '*' (line 187)
            result_mul_96509 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 36), '*', float_96507, y_96508)
            
            # Applying the binary operator '+' (line 187)
            result_add_96510 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 34), '+', result_add_96506, result_mul_96509)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'stypy_return_type', result_add_96510)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_69' in the type store
            # Getting the type of 'stypy_return_type' (line 187)
            stypy_return_type_96511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96511)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_69'
            return stypy_return_type_96511

        # Assigning a type to the variable '_stypy_temp_lambda_69' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), '_stypy_temp_lambda_69', _stypy_temp_lambda_69)
        # Getting the type of '_stypy_temp_lambda_69' (line 187)
        _stypy_temp_lambda_69_96512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 13), '_stypy_temp_lambda_69')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 13), tuple_96501, _stypy_temp_lambda_69_96512)
        # Adding element type (line 187)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_96513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        int_96514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 46), tuple_96513, int_96514)
        # Adding element type (line 187)
        float_96515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 46), tuple_96513, float_96515)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 13), tuple_96501, tuple_96513)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 16), list_96471, tuple_96501)
        
        # Assigning a type to the variable 'funcs' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'funcs', list_96471)
        
        
        # Call to enumerate(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'funcs' (line 190)
        funcs_96517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 41), 'funcs', False)
        # Processing the call keyword arguments (line 190)
        kwargs_96518 = {}
        # Getting the type of 'enumerate' (line 190)
        enumerate_96516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 190)
        enumerate_call_result_96519 = invoke(stypy.reporting.localization.Localization(__file__, 190, 31), enumerate_96516, *[funcs_96517], **kwargs_96518)
        
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), enumerate_call_result_96519)
        # Getting the type of the for loop variable (line 190)
        for_loop_var_96520 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), enumerate_call_result_96519)
        # Assigning a type to the variable 'j' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_96520))
        # Assigning a type to the variable 'func' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_96520))
        # Assigning a type to the variable 'grad' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'grad', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), for_loop_var_96520))
        # SSA begins for a for statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to func(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Obtaining the type of the subscript
        slice_96522 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 21), None, None, None)
        int_96523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 25), 'int')
        # Getting the type of 'x' (line 191)
        x_96524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___96525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 21), x_96524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_96526 = invoke(stypy.reporting.localization.Localization(__file__, 191, 21), getitem___96525, (slice_96522, int_96523))
        
        
        # Obtaining the type of the subscript
        slice_96527 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 29), None, None, None)
        int_96528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 33), 'int')
        # Getting the type of 'x' (line 191)
        x_96529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___96530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 29), x_96529, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_96531 = invoke(stypy.reporting.localization.Localization(__file__, 191, 29), getitem___96530, (slice_96527, int_96528))
        
        # Processing the call keyword arguments (line 191)
        kwargs_96532 = {}
        # Getting the type of 'func' (line 191)
        func_96521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'func', False)
        # Calling func(args, kwargs) (line 191)
        func_call_result_96533 = invoke(stypy.reporting.localization.Localization(__file__, 191, 16), func_96521, *[subscript_call_result_96526, subscript_call_result_96531], **kwargs_96532)
        
        # Assigning a type to the variable 'z' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'z', func_call_result_96533)
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to estimate_gradients_2d_global(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'tri' (line 192)
        tri_96536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 55), 'tri', False)
        # Getting the type of 'z' (line 192)
        z_96537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 60), 'z', False)
        # Processing the call keyword arguments (line 192)
        float_96538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 67), 'float')
        keyword_96539 = float_96538
        kwargs_96540 = {'tol': keyword_96539}
        # Getting the type of 'interpnd' (line 192)
        interpnd_96534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 17), 'interpnd', False)
        # Obtaining the member 'estimate_gradients_2d_global' of a type (line 192)
        estimate_gradients_2d_global_96535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 17), interpnd_96534, 'estimate_gradients_2d_global')
        # Calling estimate_gradients_2d_global(args, kwargs) (line 192)
        estimate_gradients_2d_global_call_result_96541 = invoke(stypy.reporting.localization.Localization(__file__, 192, 17), estimate_gradients_2d_global_96535, *[tri_96536, z_96537], **kwargs_96540)
        
        # Assigning a type to the variable 'dz' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'dz', estimate_gradients_2d_global_call_result_96541)
        
        # Call to assert_equal(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'dz' (line 194)
        dz_96543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 'dz', False)
        # Obtaining the member 'shape' of a type (line 194)
        shape_96544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 25), dz_96543, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_96545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        int_96546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 36), tuple_96545, int_96546)
        # Adding element type (line 194)
        int_96547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 36), tuple_96545, int_96547)
        
        # Processing the call keyword arguments (line 194)
        kwargs_96548 = {}
        # Getting the type of 'assert_equal' (line 194)
        assert_equal_96542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 194)
        assert_equal_call_result_96549 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), assert_equal_96542, *[shape_96544, tuple_96545], **kwargs_96548)
        
        
        # Call to assert_allclose(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'dz' (line 195)
        dz_96551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'dz', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 195)
        None_96552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'None', False)
        slice_96553 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 195, 32), None, None, None)
        
        # Call to array(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'grad' (line 195)
        grad_96556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 'grad', False)
        # Processing the call keyword arguments (line 195)
        kwargs_96557 = {}
        # Getting the type of 'np' (line 195)
        np_96554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 195)
        array_96555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), np_96554, 'array')
        # Calling array(args, kwargs) (line 195)
        array_call_result_96558 = invoke(stypy.reporting.localization.Localization(__file__, 195, 32), array_96555, *[grad_96556], **kwargs_96557)
        
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___96559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), array_call_result_96558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_96560 = invoke(stypy.reporting.localization.Localization(__file__, 195, 32), getitem___96559, (None_96552, slice_96553))
        
        int_96561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 57), 'int')
        # Getting the type of 'dz' (line 195)
        dz_96562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 59), 'dz', False)
        # Applying the binary operator '*' (line 195)
        result_mul_96563 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 57), '*', int_96561, dz_96562)
        
        # Applying the binary operator '+' (line 195)
        result_add_96564 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 32), '+', subscript_call_result_96560, result_mul_96563)
        
        # Processing the call keyword arguments (line 195)
        float_96565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 33), 'float')
        keyword_96566 = float_96565
        float_96567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 44), 'float')
        keyword_96568 = float_96567
        str_96569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 58), 'str', 'item %d')
        # Getting the type of 'j' (line 196)
        j_96570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 70), 'j', False)
        # Applying the binary operator '%' (line 196)
        result_mod_96571 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 58), '%', str_96569, j_96570)
        
        keyword_96572 = result_mod_96571
        kwargs_96573 = {'rtol': keyword_96566, 'err_msg': keyword_96572, 'atol': keyword_96568}
        # Getting the type of 'assert_allclose' (line 195)
        assert_allclose_96550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 195)
        assert_allclose_call_result_96574 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), assert_allclose_96550, *[dz_96551, result_add_96564], **kwargs_96573)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_96575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_smoketest'
        return stypy_return_type_96575


    @norecursion
    def test_regression_2359(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_regression_2359'
        module_type_store = module_type_store.open_function_context('test_regression_2359', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_localization', localization)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_function_name', 'TestEstimateGradients2DGlobal.test_regression_2359')
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_param_names_list', [])
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestEstimateGradients2DGlobal.test_regression_2359.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEstimateGradients2DGlobal.test_regression_2359', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_regression_2359', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_regression_2359(...)' code ##################

        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to load(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Call to data_file(...): (line 201)
        # Processing the call arguments (line 201)
        str_96579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 35), 'str', 'estimate_gradients_hang.npy')
        # Processing the call keyword arguments (line 201)
        kwargs_96580 = {}
        # Getting the type of 'data_file' (line 201)
        data_file_96578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 25), 'data_file', False)
        # Calling data_file(args, kwargs) (line 201)
        data_file_call_result_96581 = invoke(stypy.reporting.localization.Localization(__file__, 201, 25), data_file_96578, *[str_96579], **kwargs_96580)
        
        # Processing the call keyword arguments (line 201)
        kwargs_96582 = {}
        # Getting the type of 'np' (line 201)
        np_96576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 17), 'np', False)
        # Obtaining the member 'load' of a type (line 201)
        load_96577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 17), np_96576, 'load')
        # Calling load(args, kwargs) (line 201)
        load_call_result_96583 = invoke(stypy.reporting.localization.Localization(__file__, 201, 17), load_96577, *[data_file_call_result_96581], **kwargs_96582)
        
        # Assigning a type to the variable 'points' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'points', load_call_result_96583)
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to rand(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining the type of the subscript
        int_96587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'int')
        # Getting the type of 'points' (line 202)
        points_96588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'points', False)
        # Obtaining the member 'shape' of a type (line 202)
        shape_96589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), points_96588, 'shape')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___96590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 32), shape_96589, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_96591 = invoke(stypy.reporting.localization.Localization(__file__, 202, 32), getitem___96590, int_96587)
        
        # Processing the call keyword arguments (line 202)
        kwargs_96592 = {}
        # Getting the type of 'np' (line 202)
        np_96584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'np', False)
        # Obtaining the member 'random' of a type (line 202)
        random_96585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), np_96584, 'random')
        # Obtaining the member 'rand' of a type (line 202)
        rand_96586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), random_96585, 'rand')
        # Calling rand(args, kwargs) (line 202)
        rand_call_result_96593 = invoke(stypy.reporting.localization.Localization(__file__, 202, 17), rand_96586, *[subscript_call_result_96591], **kwargs_96592)
        
        # Assigning a type to the variable 'values' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'values', rand_call_result_96593)
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to Delaunay(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'points' (line 203)
        points_96596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'points', False)
        # Processing the call keyword arguments (line 203)
        kwargs_96597 = {}
        # Getting the type of 'qhull' (line 203)
        qhull_96594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 203)
        Delaunay_96595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 14), qhull_96594, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 203)
        Delaunay_call_result_96598 = invoke(stypy.reporting.localization.Localization(__file__, 203, 14), Delaunay_96595, *[points_96596], **kwargs_96597)
        
        # Assigning a type to the variable 'tri' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'tri', Delaunay_call_result_96598)
        
        # Call to suppress_warnings(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_96600 = {}
        # Getting the type of 'suppress_warnings' (line 206)
        suppress_warnings_96599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 206)
        suppress_warnings_call_result_96601 = invoke(stypy.reporting.localization.Localization(__file__, 206, 13), suppress_warnings_96599, *[], **kwargs_96600)
        
        with_96602 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 206, 13), suppress_warnings_call_result_96601, 'with parameter', '__enter__', '__exit__')

        if with_96602:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 206)
            enter___96603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 13), suppress_warnings_call_result_96601, '__enter__')
            with_enter_96604 = invoke(stypy.reporting.localization.Localization(__file__, 206, 13), enter___96603)
            # Assigning a type to the variable 'sup' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'sup', with_enter_96604)
            
            # Call to filter(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'interpnd' (line 207)
            interpnd_96607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'interpnd', False)
            # Obtaining the member 'GradientEstimationWarning' of a type (line 207)
            GradientEstimationWarning_96608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 23), interpnd_96607, 'GradientEstimationWarning')
            str_96609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'str', 'Gradient estimation did not converge')
            # Processing the call keyword arguments (line 207)
            kwargs_96610 = {}
            # Getting the type of 'sup' (line 207)
            sup_96605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 207)
            filter_96606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), sup_96605, 'filter')
            # Calling filter(args, kwargs) (line 207)
            filter_call_result_96611 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), filter_96606, *[GradientEstimationWarning_96608, str_96609], **kwargs_96610)
            
            
            # Call to estimate_gradients_2d_global(...): (line 209)
            # Processing the call arguments (line 209)
            # Getting the type of 'tri' (line 209)
            tri_96614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 50), 'tri', False)
            # Getting the type of 'values' (line 209)
            values_96615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 55), 'values', False)
            # Processing the call keyword arguments (line 209)
            int_96616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 71), 'int')
            keyword_96617 = int_96616
            kwargs_96618 = {'maxiter': keyword_96617}
            # Getting the type of 'interpnd' (line 209)
            interpnd_96612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'interpnd', False)
            # Obtaining the member 'estimate_gradients_2d_global' of a type (line 209)
            estimate_gradients_2d_global_96613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), interpnd_96612, 'estimate_gradients_2d_global')
            # Calling estimate_gradients_2d_global(args, kwargs) (line 209)
            estimate_gradients_2d_global_call_result_96619 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), estimate_gradients_2d_global_96613, *[tri_96614, values_96615], **kwargs_96618)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 206)
            exit___96620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 13), suppress_warnings_call_result_96601, '__exit__')
            with_exit_96621 = invoke(stypy.reporting.localization.Localization(__file__, 206, 13), exit___96620, None, None, None)

        
        # ################# End of 'test_regression_2359(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_regression_2359' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_96622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_regression_2359'
        return stypy_return_type_96622


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 175, 0, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestEstimateGradients2DGlobal.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestEstimateGradients2DGlobal' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'TestEstimateGradients2DGlobal', TestEstimateGradients2DGlobal)
# Declaration of the 'TestCloughTocher2DInterpolator' class

class TestCloughTocher2DInterpolator(object, ):

    @norecursion
    def _check_accuracy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 214)
        None_96623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 38), 'None')
        float_96624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'float')
        # Getting the type of 'False' (line 214)
        False_96625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 64), 'False')
        # Getting the type of 'False' (line 214)
        False_96626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 79), 'False')
        defaults = [None_96623, float_96624, False_96625, False_96626]
        # Create a new context for function '_check_accuracy'
        module_type_store = module_type_store.open_function_context('_check_accuracy', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator._check_accuracy')
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_param_names_list', ['func', 'x', 'tol', 'alternate', 'rescale'])
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator._check_accuracy.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator._check_accuracy', ['func', 'x', 'tol', 'alternate', 'rescale'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_accuracy', localization, ['func', 'x', 'tol', 'alternate', 'rescale'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_accuracy(...)' code ##################

        
        # Call to seed(...): (line 215)
        # Processing the call arguments (line 215)
        int_96630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 23), 'int')
        # Processing the call keyword arguments (line 215)
        kwargs_96631 = {}
        # Getting the type of 'np' (line 215)
        np_96627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 215)
        random_96628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), np_96627, 'random')
        # Obtaining the member 'seed' of a type (line 215)
        seed_96629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), random_96628, 'seed')
        # Calling seed(args, kwargs) (line 215)
        seed_call_result_96632 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), seed_96629, *[int_96630], **kwargs_96631)
        
        
        # Type idiom detected: calculating its left and rigth part (line 216)
        # Getting the type of 'x' (line 216)
        x_96633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'x')
        # Getting the type of 'None' (line 216)
        None_96634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'None')
        
        (may_be_96635, more_types_in_union_96636) = may_be_none(x_96633, None_96634)

        if may_be_96635:

            if more_types_in_union_96636:
                # Runtime conditional SSA (line 216)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 217):
            
            # Assigning a Call to a Name (line 217):
            
            # Call to array(...): (line 217)
            # Processing the call arguments (line 217)
            
            # Obtaining an instance of the builtin type 'list' (line 217)
            list_96639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 217)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 217)
            tuple_96640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 217)
            # Adding element type (line 217)
            int_96641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 27), tuple_96640, int_96641)
            # Adding element type (line 217)
            int_96642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 27), tuple_96640, int_96642)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96640)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 217)
            tuple_96643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 217)
            # Adding element type (line 217)
            int_96644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 35), tuple_96643, int_96644)
            # Adding element type (line 217)
            int_96645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 38), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 35), tuple_96643, int_96645)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96643)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 218)
            tuple_96646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 218)
            # Adding element type (line 218)
            int_96647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 27), tuple_96646, int_96647)
            # Adding element type (line 218)
            int_96648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 27), tuple_96646, int_96648)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96646)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 218)
            tuple_96649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 218)
            # Adding element type (line 218)
            int_96650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 35), tuple_96649, int_96650)
            # Adding element type (line 218)
            int_96651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 38), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 35), tuple_96649, int_96651)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96649)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 218)
            tuple_96652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 43), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 218)
            # Adding element type (line 218)
            float_96653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 43), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 43), tuple_96652, float_96653)
            # Adding element type (line 218)
            float_96654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 49), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 43), tuple_96652, float_96654)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96652)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 218)
            tuple_96655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 57), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 218)
            # Adding element type (line 218)
            float_96656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 57), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 57), tuple_96655, float_96656)
            # Adding element type (line 218)
            float_96657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 62), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 57), tuple_96655, float_96657)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96655)
            # Adding element type (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 219)
            tuple_96658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 219)
            # Adding element type (line 219)
            float_96659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 27), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 27), tuple_96658, float_96659)
            # Adding element type (line 219)
            float_96660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 32), 'float')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 27), tuple_96658, float_96660)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 25), list_96639, tuple_96658)
            
            # Processing the call keyword arguments (line 217)
            # Getting the type of 'float' (line 220)
            float_96661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'float', False)
            keyword_96662 = float_96661
            kwargs_96663 = {'dtype': keyword_96662}
            # Getting the type of 'np' (line 217)
            np_96637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'np', False)
            # Obtaining the member 'array' of a type (line 217)
            array_96638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), np_96637, 'array')
            # Calling array(args, kwargs) (line 217)
            array_call_result_96664 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), array_96638, *[list_96639], **kwargs_96663)
            
            # Assigning a type to the variable 'x' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'x', array_call_result_96664)

            if more_types_in_union_96636:
                # SSA join for if statement (line 216)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'alternate' (line 222)
        alternate_96665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'alternate')
        # Applying the 'not' unary operator (line 222)
        result_not__96666 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), 'not', alternate_96665)
        
        # Testing the type of an if condition (line 222)
        if_condition_96667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_not__96666)
        # Assigning a type to the variable 'if_condition_96667' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_96667', if_condition_96667)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to CloughTocher2DInterpolator(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'x' (line 223)
        x_96670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 53), 'x', False)
        
        # Call to func(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining the type of the subscript
        slice_96672 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 223, 61), None, None, None)
        int_96673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 65), 'int')
        # Getting the type of 'x' (line 223)
        x_96674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 61), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___96675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 61), x_96674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_96676 = invoke(stypy.reporting.localization.Localization(__file__, 223, 61), getitem___96675, (slice_96672, int_96673))
        
        
        # Obtaining the type of the subscript
        slice_96677 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 223, 69), None, None, None)
        int_96678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 73), 'int')
        # Getting the type of 'x' (line 223)
        x_96679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 69), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___96680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 69), x_96679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_96681 = invoke(stypy.reporting.localization.Localization(__file__, 223, 69), getitem___96680, (slice_96677, int_96678))
        
        # Processing the call keyword arguments (line 223)
        kwargs_96682 = {}
        # Getting the type of 'func' (line 223)
        func_96671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 56), 'func', False)
        # Calling func(args, kwargs) (line 223)
        func_call_result_96683 = invoke(stypy.reporting.localization.Localization(__file__, 223, 56), func_96671, *[subscript_call_result_96676, subscript_call_result_96681], **kwargs_96682)
        
        # Processing the call keyword arguments (line 223)
        float_96684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 57), 'float')
        keyword_96685 = float_96684
        # Getting the type of 'rescale' (line 224)
        rescale_96686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 71), 'rescale', False)
        keyword_96687 = rescale_96686
        kwargs_96688 = {'rescale': keyword_96687, 'tol': keyword_96685}
        # Getting the type of 'interpnd' (line 223)
        interpnd_96668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 223)
        CloughTocher2DInterpolator_96669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 17), interpnd_96668, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 223)
        CloughTocher2DInterpolator_call_result_96689 = invoke(stypy.reporting.localization.Localization(__file__, 223, 17), CloughTocher2DInterpolator_96669, *[x_96670, func_call_result_96683], **kwargs_96688)
        
        # Assigning a type to the variable 'ip' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'ip', CloughTocher2DInterpolator_call_result_96689)
        # SSA branch for the else part of an if statement (line 222)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to CloughTocher2DInterpolator(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_96692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 54), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        
        # Obtaining the type of the subscript
        slice_96693 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 54), None, None, None)
        int_96694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 58), 'int')
        # Getting the type of 'x' (line 226)
        x_96695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 54), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___96696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 54), x_96695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_96697 = invoke(stypy.reporting.localization.Localization(__file__, 226, 54), getitem___96696, (slice_96693, int_96694))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 54), tuple_96692, subscript_call_result_96697)
        # Adding element type (line 226)
        
        # Obtaining the type of the subscript
        slice_96698 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 62), None, None, None)
        int_96699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 66), 'int')
        # Getting the type of 'x' (line 226)
        x_96700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 62), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___96701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 62), x_96700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_96702 = invoke(stypy.reporting.localization.Localization(__file__, 226, 62), getitem___96701, (slice_96698, int_96699))
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 54), tuple_96692, subscript_call_result_96702)
        
        
        # Call to func(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining the type of the subscript
        slice_96704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 58), None, None, None)
        int_96705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 62), 'int')
        # Getting the type of 'x' (line 227)
        x_96706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 58), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___96707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 58), x_96706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_96708 = invoke(stypy.reporting.localization.Localization(__file__, 227, 58), getitem___96707, (slice_96704, int_96705))
        
        
        # Obtaining the type of the subscript
        slice_96709 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 66), None, None, None)
        int_96710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 70), 'int')
        # Getting the type of 'x' (line 227)
        x_96711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 66), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___96712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 66), x_96711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_96713 = invoke(stypy.reporting.localization.Localization(__file__, 227, 66), getitem___96712, (slice_96709, int_96710))
        
        # Processing the call keyword arguments (line 227)
        kwargs_96714 = {}
        # Getting the type of 'func' (line 227)
        func_96703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 53), 'func', False)
        # Calling func(args, kwargs) (line 227)
        func_call_result_96715 = invoke(stypy.reporting.localization.Localization(__file__, 227, 53), func_96703, *[subscript_call_result_96708, subscript_call_result_96713], **kwargs_96714)
        
        # Processing the call keyword arguments (line 226)
        float_96716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 57), 'float')
        keyword_96717 = float_96716
        # Getting the type of 'rescale' (line 228)
        rescale_96718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 71), 'rescale', False)
        keyword_96719 = rescale_96718
        kwargs_96720 = {'rescale': keyword_96719, 'tol': keyword_96717}
        # Getting the type of 'interpnd' (line 226)
        interpnd_96690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 226)
        CloughTocher2DInterpolator_96691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 17), interpnd_96690, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 226)
        CloughTocher2DInterpolator_call_result_96721 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), CloughTocher2DInterpolator_96691, *[tuple_96692, func_call_result_96715], **kwargs_96720)
        
        # Assigning a type to the variable 'ip' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'ip', CloughTocher2DInterpolator_call_result_96721)
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to rand(...): (line 230)
        # Processing the call arguments (line 230)
        int_96725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 27), 'int')
        int_96726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 31), 'int')
        # Processing the call keyword arguments (line 230)
        kwargs_96727 = {}
        # Getting the type of 'np' (line 230)
        np_96722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 230)
        random_96723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), np_96722, 'random')
        # Obtaining the member 'rand' of a type (line 230)
        rand_96724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), random_96723, 'rand')
        # Calling rand(args, kwargs) (line 230)
        rand_call_result_96728 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), rand_96724, *[int_96725, int_96726], **kwargs_96727)
        
        # Assigning a type to the variable 'p' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'p', rand_call_result_96728)
        
        
        # Getting the type of 'alternate' (line 232)
        alternate_96729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'alternate')
        # Applying the 'not' unary operator (line 232)
        result_not__96730 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 11), 'not', alternate_96729)
        
        # Testing the type of an if condition (line 232)
        if_condition_96731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 8), result_not__96730)
        # Assigning a type to the variable 'if_condition_96731' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'if_condition_96731', if_condition_96731)
        # SSA begins for if statement (line 232)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to ip(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'p' (line 233)
        p_96733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 19), 'p', False)
        # Processing the call keyword arguments (line 233)
        kwargs_96734 = {}
        # Getting the type of 'ip' (line 233)
        ip_96732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'ip', False)
        # Calling ip(args, kwargs) (line 233)
        ip_call_result_96735 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), ip_96732, *[p_96733], **kwargs_96734)
        
        # Assigning a type to the variable 'a' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'a', ip_call_result_96735)
        # SSA branch for the else part of an if statement (line 232)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 235):
        
        # Assigning a Call to a Name (line 235):
        
        # Call to ip(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Obtaining the type of the subscript
        slice_96737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 19), None, None, None)
        int_96738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 23), 'int')
        # Getting the type of 'p' (line 235)
        p_96739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___96740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 19), p_96739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_96741 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), getitem___96740, (slice_96737, int_96738))
        
        
        # Obtaining the type of the subscript
        slice_96742 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 27), None, None, None)
        int_96743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 31), 'int')
        # Getting the type of 'p' (line 235)
        p_96744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 27), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___96745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 27), p_96744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_96746 = invoke(stypy.reporting.localization.Localization(__file__, 235, 27), getitem___96745, (slice_96742, int_96743))
        
        # Processing the call keyword arguments (line 235)
        kwargs_96747 = {}
        # Getting the type of 'ip' (line 235)
        ip_96736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'ip', False)
        # Calling ip(args, kwargs) (line 235)
        ip_call_result_96748 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), ip_96736, *[subscript_call_result_96741, subscript_call_result_96746], **kwargs_96747)
        
        # Assigning a type to the variable 'a' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'a', ip_call_result_96748)
        # SSA join for if statement (line 232)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to func(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Obtaining the type of the subscript
        slice_96750 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 236, 17), None, None, None)
        int_96751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 21), 'int')
        # Getting the type of 'p' (line 236)
        p_96752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 17), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___96753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 17), p_96752, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_96754 = invoke(stypy.reporting.localization.Localization(__file__, 236, 17), getitem___96753, (slice_96750, int_96751))
        
        
        # Obtaining the type of the subscript
        slice_96755 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 236, 25), None, None, None)
        int_96756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'int')
        # Getting the type of 'p' (line 236)
        p_96757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___96758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 25), p_96757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_96759 = invoke(stypy.reporting.localization.Localization(__file__, 236, 25), getitem___96758, (slice_96755, int_96756))
        
        # Processing the call keyword arguments (line 236)
        kwargs_96760 = {}
        # Getting the type of 'func' (line 236)
        func_96749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'func', False)
        # Calling func(args, kwargs) (line 236)
        func_call_result_96761 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), func_96749, *[subscript_call_result_96754, subscript_call_result_96759], **kwargs_96760)
        
        # Assigning a type to the variable 'b' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'b', func_call_result_96761)
        
        
        # SSA begins for try-except statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to assert_allclose(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'a' (line 239)
        a_96763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'a', False)
        # Getting the type of 'b' (line 239)
        b_96764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'b', False)
        # Processing the call keyword arguments (line 239)
        # Getting the type of 'kw' (line 239)
        kw_96765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'kw', False)
        kwargs_96766 = {'kw_96765': kw_96765}
        # Getting the type of 'assert_allclose' (line 239)
        assert_allclose_96762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 239)
        assert_allclose_call_result_96767 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), assert_allclose_96762, *[a_96763, b_96764], **kwargs_96766)
        
        # SSA branch for the except part of a try statement (line 238)
        # SSA branch for the except 'AssertionError' branch of a try statement (line 238)
        module_type_store.open_ssa_branch('except')
        
        # Call to print(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Call to abs(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'a' (line 241)
        a_96770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'a', False)
        # Getting the type of 'b' (line 241)
        b_96771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'b', False)
        # Applying the binary operator '-' (line 241)
        result_sub_96772 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 22), '-', a_96770, b_96771)
        
        # Processing the call keyword arguments (line 241)
        kwargs_96773 = {}
        # Getting the type of 'abs' (line 241)
        abs_96769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'abs', False)
        # Calling abs(args, kwargs) (line 241)
        abs_call_result_96774 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), abs_96769, *[result_sub_96772], **kwargs_96773)
        
        # Processing the call keyword arguments (line 241)
        kwargs_96775 = {}
        # Getting the type of 'print' (line 241)
        print_96768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'print', False)
        # Calling print(args, kwargs) (line 241)
        print_call_result_96776 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), print_96768, *[abs_call_result_96774], **kwargs_96775)
        
        
        # Call to print(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'ip' (line 242)
        ip_96778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 18), 'ip', False)
        # Obtaining the member 'grad' of a type (line 242)
        grad_96779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 18), ip_96778, 'grad')
        # Processing the call keyword arguments (line 242)
        kwargs_96780 = {}
        # Getting the type of 'print' (line 242)
        print_96777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'print', False)
        # Calling print(args, kwargs) (line 242)
        print_call_result_96781 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), print_96777, *[grad_96779], **kwargs_96780)
        
        # SSA join for try-except statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_accuracy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_accuracy' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_96782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96782)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_accuracy'
        return stypy_return_type_96782


    @norecursion
    def test_linear_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear_smoketest'
        module_type_store = module_type_store.open_function_context('test_linear_smoketest', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_linear_smoketest')
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_linear_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_linear_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear_smoketest(...)' code ##################

        
        # Assigning a List to a Name (line 247):
        
        # Assigning a List to a Name (line 247):
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_96783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)

        @norecursion
        def _stypy_temp_lambda_70(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_70'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_70', 248, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_70.stypy_localization = localization
            _stypy_temp_lambda_70.stypy_type_of_self = None
            _stypy_temp_lambda_70.stypy_type_store = module_type_store
            _stypy_temp_lambda_70.stypy_function_name = '_stypy_temp_lambda_70'
            _stypy_temp_lambda_70.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_70.stypy_varargs_param_name = None
            _stypy_temp_lambda_70.stypy_kwargs_param_name = None
            _stypy_temp_lambda_70.stypy_call_defaults = defaults
            _stypy_temp_lambda_70.stypy_call_varargs = varargs
            _stypy_temp_lambda_70.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_70', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_70', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 25), 'int')
            # Getting the type of 'x' (line 248)
            x_96785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'x')
            # Applying the binary operator '*' (line 248)
            result_mul_96786 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 25), '*', int_96784, x_96785)
            
            int_96787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 31), 'int')
            # Applying the binary operator '+' (line 248)
            result_add_96788 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 25), '+', result_mul_96786, int_96787)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'stypy_return_type', result_add_96788)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_70' in the type store
            # Getting the type of 'stypy_return_type' (line 248)
            stypy_return_type_96789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96789)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_70'
            return stypy_return_type_96789

        # Assigning a type to the variable '_stypy_temp_lambda_70' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), '_stypy_temp_lambda_70', _stypy_temp_lambda_70)
        # Getting the type of '_stypy_temp_lambda_70' (line 248)
        _stypy_temp_lambda_70_96790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), '_stypy_temp_lambda_70')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), list_96783, _stypy_temp_lambda_70_96790)
        # Adding element type (line 247)

        @norecursion
        def _stypy_temp_lambda_71(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_71'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_71', 249, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_71.stypy_localization = localization
            _stypy_temp_lambda_71.stypy_type_of_self = None
            _stypy_temp_lambda_71.stypy_type_store = module_type_store
            _stypy_temp_lambda_71.stypy_function_name = '_stypy_temp_lambda_71'
            _stypy_temp_lambda_71.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_71.stypy_varargs_param_name = None
            _stypy_temp_lambda_71.stypy_kwargs_param_name = None
            _stypy_temp_lambda_71.stypy_call_defaults = defaults
            _stypy_temp_lambda_71.stypy_call_varargs = varargs
            _stypy_temp_lambda_71.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_71', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_71', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'int')
            # Getting the type of 'x' (line 249)
            x_96792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 29), 'x')
            # Applying the binary operator '+' (line 249)
            result_add_96793 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 25), '+', int_96791, x_96792)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type', result_add_96793)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_71' in the type store
            # Getting the type of 'stypy_return_type' (line 249)
            stypy_return_type_96794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96794)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_71'
            return stypy_return_type_96794

        # Assigning a type to the variable '_stypy_temp_lambda_71' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), '_stypy_temp_lambda_71', _stypy_temp_lambda_71)
        # Getting the type of '_stypy_temp_lambda_71' (line 249)
        _stypy_temp_lambda_71_96795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), '_stypy_temp_lambda_71')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), list_96783, _stypy_temp_lambda_71_96795)
        # Adding element type (line 247)

        @norecursion
        def _stypy_temp_lambda_72(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_72'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_72', 250, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_72.stypy_localization = localization
            _stypy_temp_lambda_72.stypy_type_of_self = None
            _stypy_temp_lambda_72.stypy_type_store = module_type_store
            _stypy_temp_lambda_72.stypy_function_name = '_stypy_temp_lambda_72'
            _stypy_temp_lambda_72.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_72.stypy_varargs_param_name = None
            _stypy_temp_lambda_72.stypy_kwargs_param_name = None
            _stypy_temp_lambda_72.stypy_call_defaults = defaults
            _stypy_temp_lambda_72.stypy_call_varargs = varargs
            _stypy_temp_lambda_72.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_72', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_72', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 25), 'int')
            # Getting the type of 'y' (line 250)
            y_96797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 30), 'y')
            # Applying the binary operator '+' (line 250)
            result_add_96798 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 25), '+', int_96796, y_96797)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'stypy_return_type', result_add_96798)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_72' in the type store
            # Getting the type of 'stypy_return_type' (line 250)
            stypy_return_type_96799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96799)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_72'
            return stypy_return_type_96799

        # Assigning a type to the variable '_stypy_temp_lambda_72' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), '_stypy_temp_lambda_72', _stypy_temp_lambda_72)
        # Getting the type of '_stypy_temp_lambda_72' (line 250)
        _stypy_temp_lambda_72_96800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), '_stypy_temp_lambda_72')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), list_96783, _stypy_temp_lambda_72_96800)
        # Adding element type (line 247)

        @norecursion
        def _stypy_temp_lambda_73(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_73'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_73', 251, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_73.stypy_localization = localization
            _stypy_temp_lambda_73.stypy_type_of_self = None
            _stypy_temp_lambda_73.stypy_type_store = module_type_store
            _stypy_temp_lambda_73.stypy_function_name = '_stypy_temp_lambda_73'
            _stypy_temp_lambda_73.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_73.stypy_varargs_param_name = None
            _stypy_temp_lambda_73.stypy_kwargs_param_name = None
            _stypy_temp_lambda_73.stypy_call_defaults = defaults
            _stypy_temp_lambda_73.stypy_call_varargs = varargs
            _stypy_temp_lambda_73.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_73', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_73', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_96801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'int')
            int_96802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
            # Getting the type of 'x' (line 251)
            x_96803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'x')
            # Applying the binary operator '*' (line 251)
            result_mul_96804 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 29), '*', int_96802, x_96803)
            
            # Applying the binary operator '+' (line 251)
            result_add_96805 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 25), '+', int_96801, result_mul_96804)
            
            float_96806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 35), 'float')
            # Getting the type of 'y' (line 251)
            y_96807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'y')
            # Applying the binary operator '*' (line 251)
            result_mul_96808 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 35), '*', float_96806, y_96807)
            
            # Applying the binary operator '+' (line 251)
            result_add_96809 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 33), '+', result_add_96805, result_mul_96808)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'stypy_return_type', result_add_96809)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_73' in the type store
            # Getting the type of 'stypy_return_type' (line 251)
            stypy_return_type_96810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96810)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_73'
            return stypy_return_type_96810

        # Assigning a type to the variable '_stypy_temp_lambda_73' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), '_stypy_temp_lambda_73', _stypy_temp_lambda_73)
        # Getting the type of '_stypy_temp_lambda_73' (line 251)
        _stypy_temp_lambda_73_96811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), '_stypy_temp_lambda_73')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 16), list_96783, _stypy_temp_lambda_73_96811)
        
        # Assigning a type to the variable 'funcs' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'funcs', list_96783)
        
        
        # Call to enumerate(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'funcs' (line 254)
        funcs_96813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'funcs', False)
        # Processing the call keyword arguments (line 254)
        kwargs_96814 = {}
        # Getting the type of 'enumerate' (line 254)
        enumerate_96812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 254)
        enumerate_call_result_96815 = invoke(stypy.reporting.localization.Localization(__file__, 254, 23), enumerate_96812, *[funcs_96813], **kwargs_96814)
        
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 8), enumerate_call_result_96815)
        # Getting the type of the for loop variable (line 254)
        for_loop_var_96816 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 8), enumerate_call_result_96815)
        # Assigning a type to the variable 'j' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 8), for_loop_var_96816))
        # Assigning a type to the variable 'func' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 8), for_loop_var_96816))
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_accuracy(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'func' (line 255)
        func_96819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 33), 'func', False)
        # Processing the call keyword arguments (line 255)
        float_96820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 43), 'float')
        keyword_96821 = float_96820
        float_96822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 55), 'float')
        keyword_96823 = float_96822
        float_96824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 66), 'float')
        keyword_96825 = float_96824
        str_96826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 41), 'str', 'Function %d')
        # Getting the type of 'j' (line 256)
        j_96827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 57), 'j', False)
        # Applying the binary operator '%' (line 256)
        result_mod_96828 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 41), '%', str_96826, j_96827)
        
        keyword_96829 = result_mod_96828
        kwargs_96830 = {'rtol': keyword_96825, 'err_msg': keyword_96829, 'atol': keyword_96823, 'tol': keyword_96821}
        # Getting the type of 'self' (line 255)
        self_96817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 255)
        _check_accuracy_96818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_96817, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 255)
        _check_accuracy_call_result_96831 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), _check_accuracy_96818, *[func_96819], **kwargs_96830)
        
        
        # Call to _check_accuracy(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'func' (line 257)
        func_96834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 33), 'func', False)
        # Processing the call keyword arguments (line 257)
        float_96835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 43), 'float')
        keyword_96836 = float_96835
        float_96837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 55), 'float')
        keyword_96838 = float_96837
        float_96839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 66), 'float')
        keyword_96840 = float_96839
        # Getting the type of 'True' (line 258)
        True_96841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 43), 'True', False)
        keyword_96842 = True_96841
        str_96843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 41), 'str', 'Function (alternate) %d')
        # Getting the type of 'j' (line 259)
        j_96844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 69), 'j', False)
        # Applying the binary operator '%' (line 259)
        result_mod_96845 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 41), '%', str_96843, j_96844)
        
        keyword_96846 = result_mod_96845
        kwargs_96847 = {'alternate': keyword_96842, 'rtol': keyword_96840, 'err_msg': keyword_96846, 'atol': keyword_96838, 'tol': keyword_96836}
        # Getting the type of 'self' (line 257)
        self_96832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 257)
        _check_accuracy_96833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_96832, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 257)
        _check_accuracy_call_result_96848 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), _check_accuracy_96833, *[func_96834], **kwargs_96847)
        
        
        # Call to _check_accuracy(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'func' (line 261)
        func_96851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'func', False)
        # Processing the call keyword arguments (line 261)
        float_96852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 43), 'float')
        keyword_96853 = float_96852
        float_96854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 55), 'float')
        keyword_96855 = float_96854
        float_96856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 66), 'float')
        keyword_96857 = float_96856
        str_96858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 41), 'str', 'Function (rescaled) %d')
        # Getting the type of 'j' (line 262)
        j_96859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 68), 'j', False)
        # Applying the binary operator '%' (line 262)
        result_mod_96860 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 41), '%', str_96858, j_96859)
        
        keyword_96861 = result_mod_96860
        # Getting the type of 'True' (line 262)
        True_96862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 79), 'True', False)
        keyword_96863 = True_96862
        kwargs_96864 = {'rescale': keyword_96863, 'rtol': keyword_96857, 'err_msg': keyword_96861, 'atol': keyword_96855, 'tol': keyword_96853}
        # Getting the type of 'self' (line 261)
        self_96849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 261)
        _check_accuracy_96850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), self_96849, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 261)
        _check_accuracy_call_result_96865 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), _check_accuracy_96850, *[func_96851], **kwargs_96864)
        
        
        # Call to _check_accuracy(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'func' (line 263)
        func_96868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'func', False)
        # Processing the call keyword arguments (line 263)
        float_96869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 43), 'float')
        keyword_96870 = float_96869
        float_96871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 55), 'float')
        keyword_96872 = float_96871
        float_96873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 66), 'float')
        keyword_96874 = float_96873
        # Getting the type of 'True' (line 264)
        True_96875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 43), 'True', False)
        keyword_96876 = True_96875
        # Getting the type of 'True' (line 264)
        True_96877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 57), 'True', False)
        keyword_96878 = True_96877
        str_96879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 41), 'str', 'Function (alternate, rescaled) %d')
        # Getting the type of 'j' (line 265)
        j_96880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 79), 'j', False)
        # Applying the binary operator '%' (line 265)
        result_mod_96881 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 41), '%', str_96879, j_96880)
        
        keyword_96882 = result_mod_96881
        kwargs_96883 = {'tol': keyword_96870, 'rescale': keyword_96878, 'alternate': keyword_96876, 'err_msg': keyword_96882, 'rtol': keyword_96874, 'atol': keyword_96872}
        # Getting the type of 'self' (line 263)
        self_96866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 263)
        _check_accuracy_96867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), self_96866, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 263)
        _check_accuracy_call_result_96884 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), _check_accuracy_96867, *[func_96868], **kwargs_96883)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_linear_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_96885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear_smoketest'
        return stypy_return_type_96885


    @norecursion
    def test_quadratic_smoketest(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_smoketest'
        module_type_store = module_type_store.open_function_context('test_quadratic_smoketest', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_quadratic_smoketest')
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_quadratic_smoketest.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_quadratic_smoketest', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_smoketest', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_smoketest(...)' code ##################

        
        # Assigning a List to a Name (line 269):
        
        # Assigning a List to a Name (line 269):
        
        # Obtaining an instance of the builtin type 'list' (line 269)
        list_96886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 269)
        # Adding element type (line 269)

        @norecursion
        def _stypy_temp_lambda_74(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_74'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_74', 270, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_74.stypy_localization = localization
            _stypy_temp_lambda_74.stypy_type_of_self = None
            _stypy_temp_lambda_74.stypy_type_store = module_type_store
            _stypy_temp_lambda_74.stypy_function_name = '_stypy_temp_lambda_74'
            _stypy_temp_lambda_74.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_74.stypy_varargs_param_name = None
            _stypy_temp_lambda_74.stypy_kwargs_param_name = None
            _stypy_temp_lambda_74.stypy_call_defaults = defaults
            _stypy_temp_lambda_74.stypy_call_varargs = varargs
            _stypy_temp_lambda_74.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_74', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_74', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 270)
            x_96887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'x')
            int_96888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 28), 'int')
            # Applying the binary operator '**' (line 270)
            result_pow_96889 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 25), '**', x_96887, int_96888)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stypy_return_type', result_pow_96889)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_74' in the type store
            # Getting the type of 'stypy_return_type' (line 270)
            stypy_return_type_96890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96890)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_74'
            return stypy_return_type_96890

        # Assigning a type to the variable '_stypy_temp_lambda_74' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), '_stypy_temp_lambda_74', _stypy_temp_lambda_74)
        # Getting the type of '_stypy_temp_lambda_74' (line 270)
        _stypy_temp_lambda_74_96891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), '_stypy_temp_lambda_74')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 16), list_96886, _stypy_temp_lambda_74_96891)
        # Adding element type (line 269)

        @norecursion
        def _stypy_temp_lambda_75(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_75'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_75', 271, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_75.stypy_localization = localization
            _stypy_temp_lambda_75.stypy_type_of_self = None
            _stypy_temp_lambda_75.stypy_type_store = module_type_store
            _stypy_temp_lambda_75.stypy_function_name = '_stypy_temp_lambda_75'
            _stypy_temp_lambda_75.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_75.stypy_varargs_param_name = None
            _stypy_temp_lambda_75.stypy_kwargs_param_name = None
            _stypy_temp_lambda_75.stypy_call_defaults = defaults
            _stypy_temp_lambda_75.stypy_call_varargs = varargs
            _stypy_temp_lambda_75.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_75', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_75', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'y' (line 271)
            y_96892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'y')
            int_96893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 28), 'int')
            # Applying the binary operator '**' (line 271)
            result_pow_96894 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 25), '**', y_96892, int_96893)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type', result_pow_96894)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_75' in the type store
            # Getting the type of 'stypy_return_type' (line 271)
            stypy_return_type_96895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96895)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_75'
            return stypy_return_type_96895

        # Assigning a type to the variable '_stypy_temp_lambda_75' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), '_stypy_temp_lambda_75', _stypy_temp_lambda_75)
        # Getting the type of '_stypy_temp_lambda_75' (line 271)
        _stypy_temp_lambda_75_96896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), '_stypy_temp_lambda_75')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 16), list_96886, _stypy_temp_lambda_75_96896)
        # Adding element type (line 269)

        @norecursion
        def _stypy_temp_lambda_76(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_76'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_76', 272, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_76.stypy_localization = localization
            _stypy_temp_lambda_76.stypy_type_of_self = None
            _stypy_temp_lambda_76.stypy_type_store = module_type_store
            _stypy_temp_lambda_76.stypy_function_name = '_stypy_temp_lambda_76'
            _stypy_temp_lambda_76.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_76.stypy_varargs_param_name = None
            _stypy_temp_lambda_76.stypy_kwargs_param_name = None
            _stypy_temp_lambda_76.stypy_call_defaults = defaults
            _stypy_temp_lambda_76.stypy_call_varargs = varargs
            _stypy_temp_lambda_76.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_76', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_76', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 272)
            x_96897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'x')
            int_96898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 28), 'int')
            # Applying the binary operator '**' (line 272)
            result_pow_96899 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 25), '**', x_96897, int_96898)
            
            # Getting the type of 'y' (line 272)
            y_96900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'y')
            int_96901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 35), 'int')
            # Applying the binary operator '**' (line 272)
            result_pow_96902 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 32), '**', y_96900, int_96901)
            
            # Applying the binary operator '-' (line 272)
            result_sub_96903 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 25), '-', result_pow_96899, result_pow_96902)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type', result_sub_96903)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_76' in the type store
            # Getting the type of 'stypy_return_type' (line 272)
            stypy_return_type_96904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96904)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_76'
            return stypy_return_type_96904

        # Assigning a type to the variable '_stypy_temp_lambda_76' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), '_stypy_temp_lambda_76', _stypy_temp_lambda_76)
        # Getting the type of '_stypy_temp_lambda_76' (line 272)
        _stypy_temp_lambda_76_96905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), '_stypy_temp_lambda_76')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 16), list_96886, _stypy_temp_lambda_76_96905)
        # Adding element type (line 269)

        @norecursion
        def _stypy_temp_lambda_77(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_77'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_77', 273, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_77.stypy_localization = localization
            _stypy_temp_lambda_77.stypy_type_of_self = None
            _stypy_temp_lambda_77.stypy_type_store = module_type_store
            _stypy_temp_lambda_77.stypy_function_name = '_stypy_temp_lambda_77'
            _stypy_temp_lambda_77.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_77.stypy_varargs_param_name = None
            _stypy_temp_lambda_77.stypy_kwargs_param_name = None
            _stypy_temp_lambda_77.stypy_call_defaults = defaults
            _stypy_temp_lambda_77.stypy_call_varargs = varargs
            _stypy_temp_lambda_77.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_77', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_77', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 273)
            x_96906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'x')
            # Getting the type of 'y' (line 273)
            y_96907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 27), 'y')
            # Applying the binary operator '*' (line 273)
            result_mul_96908 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 25), '*', x_96906, y_96907)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'stypy_return_type', result_mul_96908)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_77' in the type store
            # Getting the type of 'stypy_return_type' (line 273)
            stypy_return_type_96909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_96909)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_77'
            return stypy_return_type_96909

        # Assigning a type to the variable '_stypy_temp_lambda_77' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), '_stypy_temp_lambda_77', _stypy_temp_lambda_77)
        # Getting the type of '_stypy_temp_lambda_77' (line 273)
        _stypy_temp_lambda_77_96910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), '_stypy_temp_lambda_77')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 16), list_96886, _stypy_temp_lambda_77_96910)
        
        # Assigning a type to the variable 'funcs' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'funcs', list_96886)
        
        
        # Call to enumerate(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'funcs' (line 276)
        funcs_96912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 33), 'funcs', False)
        # Processing the call keyword arguments (line 276)
        kwargs_96913 = {}
        # Getting the type of 'enumerate' (line 276)
        enumerate_96911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 276)
        enumerate_call_result_96914 = invoke(stypy.reporting.localization.Localization(__file__, 276, 23), enumerate_96911, *[funcs_96912], **kwargs_96913)
        
        # Testing the type of a for loop iterable (line 276)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 276, 8), enumerate_call_result_96914)
        # Getting the type of the for loop variable (line 276)
        for_loop_var_96915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 276, 8), enumerate_call_result_96914)
        # Assigning a type to the variable 'j' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 8), for_loop_var_96915))
        # Assigning a type to the variable 'func' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 8), for_loop_var_96915))
        # SSA begins for a for statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_accuracy(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'func' (line 277)
        func_96918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 33), 'func', False)
        # Processing the call keyword arguments (line 277)
        float_96919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 43), 'float')
        keyword_96920 = float_96919
        float_96921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 54), 'float')
        keyword_96922 = float_96921
        int_96923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 65), 'int')
        keyword_96924 = int_96923
        str_96925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 41), 'str', 'Function %d')
        # Getting the type of 'j' (line 278)
        j_96926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 57), 'j', False)
        # Applying the binary operator '%' (line 278)
        result_mod_96927 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 41), '%', str_96925, j_96926)
        
        keyword_96928 = result_mod_96927
        kwargs_96929 = {'rtol': keyword_96924, 'err_msg': keyword_96928, 'atol': keyword_96922, 'tol': keyword_96920}
        # Getting the type of 'self' (line 277)
        self_96916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 277)
        _check_accuracy_96917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), self_96916, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 277)
        _check_accuracy_call_result_96930 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), _check_accuracy_96917, *[func_96918], **kwargs_96929)
        
        
        # Call to _check_accuracy(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'func' (line 279)
        func_96933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'func', False)
        # Processing the call keyword arguments (line 279)
        float_96934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 43), 'float')
        keyword_96935 = float_96934
        float_96936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 54), 'float')
        keyword_96937 = float_96936
        int_96938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 65), 'int')
        keyword_96939 = int_96938
        str_96940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 41), 'str', 'Function %d')
        # Getting the type of 'j' (line 280)
        j_96941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 57), 'j', False)
        # Applying the binary operator '%' (line 280)
        result_mod_96942 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 41), '%', str_96940, j_96941)
        
        keyword_96943 = result_mod_96942
        # Getting the type of 'True' (line 280)
        True_96944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 68), 'True', False)
        keyword_96945 = True_96944
        kwargs_96946 = {'rescale': keyword_96945, 'rtol': keyword_96939, 'err_msg': keyword_96943, 'atol': keyword_96937, 'tol': keyword_96935}
        # Getting the type of 'self' (line 279)
        self_96931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 279)
        _check_accuracy_96932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), self_96931, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 279)
        _check_accuracy_call_result_96947 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), _check_accuracy_96932, *[func_96933], **kwargs_96946)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_quadratic_smoketest(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_smoketest' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_96948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_96948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_smoketest'
        return stypy_return_type_96948


    @norecursion
    def test_tri_input(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tri_input'
        module_type_store = module_type_store.open_function_context('test_tri_input', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_tri_input')
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_tri_input.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_tri_input', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tri_input', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tri_input(...)' code ##################

        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to array(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Obtaining an instance of the builtin type 'list' (line 284)
        list_96951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 284)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_96952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        int_96953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 23), tuple_96952, int_96953)
        # Adding element type (line 284)
        int_96954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 23), tuple_96952, int_96954)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 21), list_96951, tuple_96952)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_96955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        float_96956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 30), tuple_96955, float_96956)
        # Adding element type (line 284)
        float_96957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 30), tuple_96955, float_96957)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 21), list_96951, tuple_96955)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_96958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        float_96959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 43), tuple_96958, float_96959)
        # Adding element type (line 284)
        float_96960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 43), tuple_96958, float_96960)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 21), list_96951, tuple_96958)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_96961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        float_96962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 55), tuple_96961, float_96962)
        # Adding element type (line 284)
        float_96963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 55), tuple_96961, float_96963)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 21), list_96951, tuple_96961)
        # Adding element type (line 284)
        
        # Obtaining an instance of the builtin type 'tuple' (line 284)
        tuple_96964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 284)
        # Adding element type (line 284)
        float_96965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 67), tuple_96964, float_96965)
        # Adding element type (line 284)
        float_96966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 73), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 67), tuple_96964, float_96966)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 21), list_96951, tuple_96964)
        
        # Processing the call keyword arguments (line 284)
        # Getting the type of 'np' (line 285)
        np_96967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 285)
        double_96968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 27), np_96967, 'double')
        keyword_96969 = double_96968
        kwargs_96970 = {'dtype': keyword_96969}
        # Getting the type of 'np' (line 284)
        np_96949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 284)
        array_96950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), np_96949, 'array')
        # Calling array(args, kwargs) (line 284)
        array_call_result_96971 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), array_96950, *[list_96951], **kwargs_96970)
        
        # Assigning a type to the variable 'x' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'x', array_call_result_96971)
        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to arange(...): (line 286)
        # Processing the call arguments (line 286)
        
        # Obtaining the type of the subscript
        int_96974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 30), 'int')
        # Getting the type of 'x' (line 286)
        x_96975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 286)
        shape_96976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 22), x_96975, 'shape')
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___96977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 22), shape_96976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_96978 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), getitem___96977, int_96974)
        
        # Processing the call keyword arguments (line 286)
        # Getting the type of 'np' (line 286)
        np_96979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 286)
        double_96980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 40), np_96979, 'double')
        keyword_96981 = double_96980
        kwargs_96982 = {'dtype': keyword_96981}
        # Getting the type of 'np' (line 286)
        np_96972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 286)
        arange_96973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), np_96972, 'arange')
        # Calling arange(args, kwargs) (line 286)
        arange_call_result_96983 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), arange_96973, *[subscript_call_result_96978], **kwargs_96982)
        
        # Assigning a type to the variable 'y' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'y', arange_call_result_96983)
        
        # Assigning a BinOp to a Name (line 287):
        
        # Assigning a BinOp to a Name (line 287):
        # Getting the type of 'y' (line 287)
        y_96984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'y')
        complex_96985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 16), 'complex')
        # Getting the type of 'y' (line 287)
        y_96986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'y')
        # Applying the binary operator '*' (line 287)
        result_mul_96987 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 16), '*', complex_96985, y_96986)
        
        # Applying the binary operator '-' (line 287)
        result_sub_96988 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 12), '-', y_96984, result_mul_96987)
        
        # Assigning a type to the variable 'y' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'y', result_sub_96988)
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to Delaunay(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'x' (line 289)
        x_96991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'x', False)
        # Processing the call keyword arguments (line 289)
        kwargs_96992 = {}
        # Getting the type of 'qhull' (line 289)
        qhull_96989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 289)
        Delaunay_96990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 14), qhull_96989, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 289)
        Delaunay_call_result_96993 = invoke(stypy.reporting.localization.Localization(__file__, 289, 14), Delaunay_96990, *[x_96991], **kwargs_96992)
        
        # Assigning a type to the variable 'tri' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tri', Delaunay_call_result_96993)
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to (...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'x' (line 290)
        x_97000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 57), 'x', False)
        # Processing the call keyword arguments (line 290)
        kwargs_97001 = {}
        
        # Call to CloughTocher2DInterpolator(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'tri' (line 290)
        tri_96996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 49), 'tri', False)
        # Getting the type of 'y' (line 290)
        y_96997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 54), 'y', False)
        # Processing the call keyword arguments (line 290)
        kwargs_96998 = {}
        # Getting the type of 'interpnd' (line 290)
        interpnd_96994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 13), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 290)
        CloughTocher2DInterpolator_96995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 13), interpnd_96994, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 290)
        CloughTocher2DInterpolator_call_result_96999 = invoke(stypy.reporting.localization.Localization(__file__, 290, 13), CloughTocher2DInterpolator_96995, *[tri_96996, y_96997], **kwargs_96998)
        
        # Calling (args, kwargs) (line 290)
        _call_result_97002 = invoke(stypy.reporting.localization.Localization(__file__, 290, 13), CloughTocher2DInterpolator_call_result_96999, *[x_97000], **kwargs_97001)
        
        # Assigning a type to the variable 'yi' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'yi', _call_result_97002)
        
        # Call to assert_almost_equal(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'y' (line 291)
        y_97004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'y', False)
        # Getting the type of 'yi' (line 291)
        yi_97005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'yi', False)
        # Processing the call keyword arguments (line 291)
        kwargs_97006 = {}
        # Getting the type of 'assert_almost_equal' (line 291)
        assert_almost_equal_97003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 291)
        assert_almost_equal_call_result_97007 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), assert_almost_equal_97003, *[y_97004, yi_97005], **kwargs_97006)
        
        
        # ################# End of 'test_tri_input(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tri_input' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_97008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tri_input'
        return stypy_return_type_97008


    @norecursion
    def test_tri_input_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tri_input_rescale'
        module_type_store = module_type_store.open_function_context('test_tri_input_rescale', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_tri_input_rescale')
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_tri_input_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_tri_input_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tri_input_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tri_input_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to array(...): (line 295)
        # Processing the call arguments (line 295)
        
        # Obtaining an instance of the builtin type 'list' (line 295)
        list_97011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 295)
        # Adding element type (line 295)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_97012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_97013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 23), tuple_97012, int_97013)
        # Adding element type (line 295)
        int_97014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 23), tuple_97012, int_97014)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_97011, tuple_97012)
        # Adding element type (line 295)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_97015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_97016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 30), tuple_97015, int_97016)
        # Adding element type (line 295)
        int_97017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 30), tuple_97015, int_97017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_97011, tuple_97015)
        # Adding element type (line 295)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_97018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_97019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 39), tuple_97018, int_97019)
        # Adding element type (line 295)
        int_97020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 39), tuple_97018, int_97020)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_97011, tuple_97018)
        # Adding element type (line 295)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_97021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_97022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 47), tuple_97021, int_97022)
        # Adding element type (line 295)
        int_97023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 47), tuple_97021, int_97023)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_97011, tuple_97021)
        # Adding element type (line 295)
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_97024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        float_97025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 55), tuple_97024, float_97025)
        # Adding element type (line 295)
        int_97026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 55), tuple_97024, int_97026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 21), list_97011, tuple_97024)
        
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'np' (line 296)
        np_97027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 296)
        double_97028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), np_97027, 'double')
        keyword_97029 = double_97028
        kwargs_97030 = {'dtype': keyword_97029}
        # Getting the type of 'np' (line 295)
        np_97009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 295)
        array_97010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), np_97009, 'array')
        # Calling array(args, kwargs) (line 295)
        array_call_result_97031 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), array_97010, *[list_97011], **kwargs_97030)
        
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'x', array_call_result_97031)
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to arange(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Obtaining the type of the subscript
        int_97034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 30), 'int')
        # Getting the type of 'x' (line 297)
        x_97035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 297)
        shape_97036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 22), x_97035, 'shape')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___97037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 22), shape_97036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_97038 = invoke(stypy.reporting.localization.Localization(__file__, 297, 22), getitem___97037, int_97034)
        
        # Processing the call keyword arguments (line 297)
        # Getting the type of 'np' (line 297)
        np_97039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 297)
        double_97040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 40), np_97039, 'double')
        keyword_97041 = double_97040
        kwargs_97042 = {'dtype': keyword_97041}
        # Getting the type of 'np' (line 297)
        np_97032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 297)
        arange_97033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), np_97032, 'arange')
        # Calling arange(args, kwargs) (line 297)
        arange_call_result_97043 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), arange_97033, *[subscript_call_result_97038], **kwargs_97042)
        
        # Assigning a type to the variable 'y' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'y', arange_call_result_97043)
        
        # Assigning a BinOp to a Name (line 298):
        
        # Assigning a BinOp to a Name (line 298):
        # Getting the type of 'y' (line 298)
        y_97044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'y')
        complex_97045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 16), 'complex')
        # Getting the type of 'y' (line 298)
        y_97046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'y')
        # Applying the binary operator '*' (line 298)
        result_mul_97047 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 16), '*', complex_97045, y_97046)
        
        # Applying the binary operator '-' (line 298)
        result_sub_97048 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 12), '-', y_97044, result_mul_97047)
        
        # Assigning a type to the variable 'y' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'y', result_sub_97048)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to Delaunay(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'x' (line 300)
        x_97051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 'x', False)
        # Processing the call keyword arguments (line 300)
        kwargs_97052 = {}
        # Getting the type of 'qhull' (line 300)
        qhull_97049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 300)
        Delaunay_97050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 14), qhull_97049, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 300)
        Delaunay_call_result_97053 = invoke(stypy.reporting.localization.Localization(__file__, 300, 14), Delaunay_97050, *[x_97051], **kwargs_97052)
        
        # Assigning a type to the variable 'tri' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tri', Delaunay_call_result_97053)
        
        
        # SSA begins for try-except statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to (...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'x' (line 302)
        x_97062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 70), 'x', False)
        # Processing the call keyword arguments (line 302)
        kwargs_97063 = {}
        
        # Call to CloughTocher2DInterpolator(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'tri' (line 302)
        tri_97056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 48), 'tri', False)
        # Getting the type of 'y' (line 302)
        y_97057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 53), 'y', False)
        # Processing the call keyword arguments (line 302)
        # Getting the type of 'True' (line 302)
        True_97058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 64), 'True', False)
        keyword_97059 = True_97058
        kwargs_97060 = {'rescale': keyword_97059}
        # Getting the type of 'interpnd' (line 302)
        interpnd_97054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 302)
        CloughTocher2DInterpolator_97055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), interpnd_97054, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 302)
        CloughTocher2DInterpolator_call_result_97061 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), CloughTocher2DInterpolator_97055, *[tri_97056, y_97057], **kwargs_97060)
        
        # Calling (args, kwargs) (line 302)
        _call_result_97064 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), CloughTocher2DInterpolator_call_result_97061, *[x_97062], **kwargs_97063)
        
        # SSA branch for the except part of a try statement (line 301)
        # SSA branch for the except 'ValueError' branch of a try statement (line 301)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'ValueError' (line 303)
        ValueError_97065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'ValueError')
        # Assigning a type to the variable 'a' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'a', ValueError_97065)
        
        
        
        # Call to str(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'a' (line 304)
        a_97067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'a', False)
        # Processing the call keyword arguments (line 304)
        kwargs_97068 = {}
        # Getting the type of 'str' (line 304)
        str_97066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'str', False)
        # Calling str(args, kwargs) (line 304)
        str_call_result_97069 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), str_97066, *[a_97067], **kwargs_97068)
        
        str_97070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 26), 'str', 'Rescaling is not supported when passing a Delaunay triangulation as ``points``.')
        # Applying the binary operator '!=' (line 304)
        result_ne_97071 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 15), '!=', str_call_result_97069, str_97070)
        
        # Testing the type of an if condition (line 304)
        if_condition_97072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 12), result_ne_97071)
        # Assigning a type to the variable 'if_condition_97072' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'if_condition_97072', if_condition_97072)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except '<any exception>' branch of a try statement (line 301)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tri_input_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tri_input_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_97073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97073)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tri_input_rescale'
        return stypy_return_type_97073


    @norecursion
    def test_tripoints_input_rescale(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tripoints_input_rescale'
        module_type_store = module_type_store.open_function_context('test_tripoints_input_rescale', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_tripoints_input_rescale')
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_tripoints_input_rescale.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_tripoints_input_rescale', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tripoints_input_rescale', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tripoints_input_rescale(...)' code ##################

        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to array(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_97076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        # Adding element type (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_97077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        int_97078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 23), tuple_97077, int_97078)
        # Adding element type (line 312)
        int_97079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 23), tuple_97077, int_97079)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 21), list_97076, tuple_97077)
        # Adding element type (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_97080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        int_97081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 30), tuple_97080, int_97081)
        # Adding element type (line 312)
        int_97082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 30), tuple_97080, int_97082)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 21), list_97076, tuple_97080)
        # Adding element type (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_97083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        int_97084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 39), tuple_97083, int_97084)
        # Adding element type (line 312)
        int_97085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 39), tuple_97083, int_97085)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 21), list_97076, tuple_97083)
        # Adding element type (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_97086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        int_97087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 47), tuple_97086, int_97087)
        # Adding element type (line 312)
        int_97088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 47), tuple_97086, int_97088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 21), list_97076, tuple_97086)
        # Adding element type (line 312)
        
        # Obtaining an instance of the builtin type 'tuple' (line 312)
        tuple_97089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 312)
        # Adding element type (line 312)
        float_97090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 55), tuple_97089, float_97090)
        # Adding element type (line 312)
        int_97091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 55), tuple_97089, int_97091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 312, 21), list_97076, tuple_97089)
        
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'np' (line 313)
        np_97092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 27), 'np', False)
        # Obtaining the member 'double' of a type (line 313)
        double_97093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 27), np_97092, 'double')
        keyword_97094 = double_97093
        kwargs_97095 = {'dtype': keyword_97094}
        # Getting the type of 'np' (line 312)
        np_97074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 312)
        array_97075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), np_97074, 'array')
        # Calling array(args, kwargs) (line 312)
        array_call_result_97096 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), array_97075, *[list_97076], **kwargs_97095)
        
        # Assigning a type to the variable 'x' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'x', array_call_result_97096)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to arange(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Obtaining the type of the subscript
        int_97099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 30), 'int')
        # Getting the type of 'x' (line 314)
        x_97100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'x', False)
        # Obtaining the member 'shape' of a type (line 314)
        shape_97101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 22), x_97100, 'shape')
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___97102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 22), shape_97101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_97103 = invoke(stypy.reporting.localization.Localization(__file__, 314, 22), getitem___97102, int_97099)
        
        # Processing the call keyword arguments (line 314)
        # Getting the type of 'np' (line 314)
        np_97104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 40), 'np', False)
        # Obtaining the member 'double' of a type (line 314)
        double_97105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 40), np_97104, 'double')
        keyword_97106 = double_97105
        kwargs_97107 = {'dtype': keyword_97106}
        # Getting the type of 'np' (line 314)
        np_97097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 314)
        arange_97098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), np_97097, 'arange')
        # Calling arange(args, kwargs) (line 314)
        arange_call_result_97108 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), arange_97098, *[subscript_call_result_97103], **kwargs_97107)
        
        # Assigning a type to the variable 'y' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'y', arange_call_result_97108)
        
        # Assigning a BinOp to a Name (line 315):
        
        # Assigning a BinOp to a Name (line 315):
        # Getting the type of 'y' (line 315)
        y_97109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'y')
        complex_97110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 16), 'complex')
        # Getting the type of 'y' (line 315)
        y_97111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'y')
        # Applying the binary operator '*' (line 315)
        result_mul_97112 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 16), '*', complex_97110, y_97111)
        
        # Applying the binary operator '-' (line 315)
        result_sub_97113 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 12), '-', y_97109, result_mul_97112)
        
        # Assigning a type to the variable 'y' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'y', result_sub_97113)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to Delaunay(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'x' (line 317)
        x_97116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 29), 'x', False)
        # Processing the call keyword arguments (line 317)
        kwargs_97117 = {}
        # Getting the type of 'qhull' (line 317)
        qhull_97114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 14), 'qhull', False)
        # Obtaining the member 'Delaunay' of a type (line 317)
        Delaunay_97115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 14), qhull_97114, 'Delaunay')
        # Calling Delaunay(args, kwargs) (line 317)
        Delaunay_call_result_97118 = invoke(stypy.reporting.localization.Localization(__file__, 317, 14), Delaunay_97115, *[x_97116], **kwargs_97117)
        
        # Assigning a type to the variable 'tri' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'tri', Delaunay_call_result_97118)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to (...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'x' (line 318)
        x_97126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 64), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_97127 = {}
        
        # Call to CloughTocher2DInterpolator(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'tri' (line 318)
        tri_97121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 49), 'tri', False)
        # Obtaining the member 'points' of a type (line 318)
        points_97122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 49), tri_97121, 'points')
        # Getting the type of 'y' (line 318)
        y_97123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 61), 'y', False)
        # Processing the call keyword arguments (line 318)
        kwargs_97124 = {}
        # Getting the type of 'interpnd' (line 318)
        interpnd_97119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 13), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 318)
        CloughTocher2DInterpolator_97120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 13), interpnd_97119, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 318)
        CloughTocher2DInterpolator_call_result_97125 = invoke(stypy.reporting.localization.Localization(__file__, 318, 13), CloughTocher2DInterpolator_97120, *[points_97122, y_97123], **kwargs_97124)
        
        # Calling (args, kwargs) (line 318)
        _call_result_97128 = invoke(stypy.reporting.localization.Localization(__file__, 318, 13), CloughTocher2DInterpolator_call_result_97125, *[x_97126], **kwargs_97127)
        
        # Assigning a type to the variable 'yi' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'yi', _call_result_97128)
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to (...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'x' (line 319)
        x_97138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 86), 'x', False)
        # Processing the call keyword arguments (line 319)
        kwargs_97139 = {}
        
        # Call to CloughTocher2DInterpolator(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'tri' (line 319)
        tri_97131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 57), 'tri', False)
        # Obtaining the member 'points' of a type (line 319)
        points_97132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 57), tri_97131, 'points')
        # Getting the type of 'y' (line 319)
        y_97133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 69), 'y', False)
        # Processing the call keyword arguments (line 319)
        # Getting the type of 'True' (line 319)
        True_97134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 80), 'True', False)
        keyword_97135 = True_97134
        kwargs_97136 = {'rescale': keyword_97135}
        # Getting the type of 'interpnd' (line 319)
        interpnd_97129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 319)
        CloughTocher2DInterpolator_97130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 21), interpnd_97129, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 319)
        CloughTocher2DInterpolator_call_result_97137 = invoke(stypy.reporting.localization.Localization(__file__, 319, 21), CloughTocher2DInterpolator_97130, *[points_97132, y_97133], **kwargs_97136)
        
        # Calling (args, kwargs) (line 319)
        _call_result_97140 = invoke(stypy.reporting.localization.Localization(__file__, 319, 21), CloughTocher2DInterpolator_call_result_97137, *[x_97138], **kwargs_97139)
        
        # Assigning a type to the variable 'yi_rescale' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'yi_rescale', _call_result_97140)
        
        # Call to assert_almost_equal(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'yi' (line 320)
        yi_97142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 28), 'yi', False)
        # Getting the type of 'yi_rescale' (line 320)
        yi_rescale_97143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 32), 'yi_rescale', False)
        # Processing the call keyword arguments (line 320)
        kwargs_97144 = {}
        # Getting the type of 'assert_almost_equal' (line 320)
        assert_almost_equal_97141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 320)
        assert_almost_equal_call_result_97145 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), assert_almost_equal_97141, *[yi_97142, yi_rescale_97143], **kwargs_97144)
        
        
        # ################# End of 'test_tripoints_input_rescale(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tripoints_input_rescale' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_97146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tripoints_input_rescale'
        return stypy_return_type_97146


    @norecursion
    def test_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dense'
        module_type_store = module_type_store.open_function_context('test_dense', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_dense')
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_dense.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_dense', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dense', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dense(...)' code ##################

        
        # Assigning a List to a Name (line 324):
        
        # Assigning a List to a Name (line 324):
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_97147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)

        @norecursion
        def _stypy_temp_lambda_78(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_78'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_78', 325, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_78.stypy_localization = localization
            _stypy_temp_lambda_78.stypy_type_of_self = None
            _stypy_temp_lambda_78.stypy_type_store = module_type_store
            _stypy_temp_lambda_78.stypy_function_name = '_stypy_temp_lambda_78'
            _stypy_temp_lambda_78.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_78.stypy_varargs_param_name = None
            _stypy_temp_lambda_78.stypy_kwargs_param_name = None
            _stypy_temp_lambda_78.stypy_call_defaults = defaults
            _stypy_temp_lambda_78.stypy_call_varargs = varargs
            _stypy_temp_lambda_78.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_78', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_78', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 325)
            x_97148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 25), 'x')
            int_97149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'int')
            # Applying the binary operator '**' (line 325)
            result_pow_97150 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 25), '**', x_97148, int_97149)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'stypy_return_type', result_pow_97150)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_78' in the type store
            # Getting the type of 'stypy_return_type' (line 325)
            stypy_return_type_97151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_97151)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_78'
            return stypy_return_type_97151

        # Assigning a type to the variable '_stypy_temp_lambda_78' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), '_stypy_temp_lambda_78', _stypy_temp_lambda_78)
        # Getting the type of '_stypy_temp_lambda_78' (line 325)
        _stypy_temp_lambda_78_97152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), '_stypy_temp_lambda_78')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 16), list_97147, _stypy_temp_lambda_78_97152)
        # Adding element type (line 324)

        @norecursion
        def _stypy_temp_lambda_79(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_79'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_79', 326, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_79.stypy_localization = localization
            _stypy_temp_lambda_79.stypy_type_of_self = None
            _stypy_temp_lambda_79.stypy_type_store = module_type_store
            _stypy_temp_lambda_79.stypy_function_name = '_stypy_temp_lambda_79'
            _stypy_temp_lambda_79.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_79.stypy_varargs_param_name = None
            _stypy_temp_lambda_79.stypy_kwargs_param_name = None
            _stypy_temp_lambda_79.stypy_call_defaults = defaults
            _stypy_temp_lambda_79.stypy_call_varargs = varargs
            _stypy_temp_lambda_79.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_79', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_79', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'y' (line 326)
            y_97153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'y')
            int_97154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 28), 'int')
            # Applying the binary operator '**' (line 326)
            result_pow_97155 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 25), '**', y_97153, int_97154)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type', result_pow_97155)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_79' in the type store
            # Getting the type of 'stypy_return_type' (line 326)
            stypy_return_type_97156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_97156)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_79'
            return stypy_return_type_97156

        # Assigning a type to the variable '_stypy_temp_lambda_79' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), '_stypy_temp_lambda_79', _stypy_temp_lambda_79)
        # Getting the type of '_stypy_temp_lambda_79' (line 326)
        _stypy_temp_lambda_79_97157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), '_stypy_temp_lambda_79')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 16), list_97147, _stypy_temp_lambda_79_97157)
        # Adding element type (line 324)

        @norecursion
        def _stypy_temp_lambda_80(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_80'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_80', 327, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_80.stypy_localization = localization
            _stypy_temp_lambda_80.stypy_type_of_self = None
            _stypy_temp_lambda_80.stypy_type_store = module_type_store
            _stypy_temp_lambda_80.stypy_function_name = '_stypy_temp_lambda_80'
            _stypy_temp_lambda_80.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_80.stypy_varargs_param_name = None
            _stypy_temp_lambda_80.stypy_kwargs_param_name = None
            _stypy_temp_lambda_80.stypy_call_defaults = defaults
            _stypy_temp_lambda_80.stypy_call_varargs = varargs
            _stypy_temp_lambda_80.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_80', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_80', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 327)
            x_97158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 25), 'x')
            int_97159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 28), 'int')
            # Applying the binary operator '**' (line 327)
            result_pow_97160 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 25), '**', x_97158, int_97159)
            
            # Getting the type of 'y' (line 327)
            y_97161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 32), 'y')
            int_97162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 35), 'int')
            # Applying the binary operator '**' (line 327)
            result_pow_97163 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 32), '**', y_97161, int_97162)
            
            # Applying the binary operator '-' (line 327)
            result_sub_97164 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 25), '-', result_pow_97160, result_pow_97163)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'stypy_return_type', result_sub_97164)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_80' in the type store
            # Getting the type of 'stypy_return_type' (line 327)
            stypy_return_type_97165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_97165)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_80'
            return stypy_return_type_97165

        # Assigning a type to the variable '_stypy_temp_lambda_80' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), '_stypy_temp_lambda_80', _stypy_temp_lambda_80)
        # Getting the type of '_stypy_temp_lambda_80' (line 327)
        _stypy_temp_lambda_80_97166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), '_stypy_temp_lambda_80')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 16), list_97147, _stypy_temp_lambda_80_97166)
        # Adding element type (line 324)

        @norecursion
        def _stypy_temp_lambda_81(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_81'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_81', 328, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_81.stypy_localization = localization
            _stypy_temp_lambda_81.stypy_type_of_self = None
            _stypy_temp_lambda_81.stypy_type_store = module_type_store
            _stypy_temp_lambda_81.stypy_function_name = '_stypy_temp_lambda_81'
            _stypy_temp_lambda_81.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_81.stypy_varargs_param_name = None
            _stypy_temp_lambda_81.stypy_kwargs_param_name = None
            _stypy_temp_lambda_81.stypy_call_defaults = defaults
            _stypy_temp_lambda_81.stypy_call_varargs = varargs
            _stypy_temp_lambda_81.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_81', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_81', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 328)
            x_97167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 25), 'x')
            # Getting the type of 'y' (line 328)
            y_97168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'y')
            # Applying the binary operator '*' (line 328)
            result_mul_97169 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 25), '*', x_97167, y_97168)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'stypy_return_type', result_mul_97169)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_81' in the type store
            # Getting the type of 'stypy_return_type' (line 328)
            stypy_return_type_97170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_97170)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_81'
            return stypy_return_type_97170

        # Assigning a type to the variable '_stypy_temp_lambda_81' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), '_stypy_temp_lambda_81', _stypy_temp_lambda_81)
        # Getting the type of '_stypy_temp_lambda_81' (line 328)
        _stypy_temp_lambda_81_97171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), '_stypy_temp_lambda_81')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 16), list_97147, _stypy_temp_lambda_81_97171)
        # Adding element type (line 324)

        @norecursion
        def _stypy_temp_lambda_82(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_82'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_82', 329, 12, True)
            # Passed parameters checking function
            _stypy_temp_lambda_82.stypy_localization = localization
            _stypy_temp_lambda_82.stypy_type_of_self = None
            _stypy_temp_lambda_82.stypy_type_store = module_type_store
            _stypy_temp_lambda_82.stypy_function_name = '_stypy_temp_lambda_82'
            _stypy_temp_lambda_82.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_82.stypy_varargs_param_name = None
            _stypy_temp_lambda_82.stypy_kwargs_param_name = None
            _stypy_temp_lambda_82.stypy_call_defaults = defaults
            _stypy_temp_lambda_82.stypy_call_varargs = varargs
            _stypy_temp_lambda_82.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_82', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_82', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to cos(...): (line 329)
            # Processing the call arguments (line 329)
            int_97174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 32), 'int')
            # Getting the type of 'np' (line 329)
            np_97175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'np', False)
            # Obtaining the member 'pi' of a type (line 329)
            pi_97176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 34), np_97175, 'pi')
            # Applying the binary operator '*' (line 329)
            result_mul_97177 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 32), '*', int_97174, pi_97176)
            
            # Getting the type of 'x' (line 329)
            x_97178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 40), 'x', False)
            # Applying the binary operator '*' (line 329)
            result_mul_97179 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 39), '*', result_mul_97177, x_97178)
            
            # Processing the call keyword arguments (line 329)
            kwargs_97180 = {}
            # Getting the type of 'np' (line 329)
            np_97172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 25), 'np', False)
            # Obtaining the member 'cos' of a type (line 329)
            cos_97173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 25), np_97172, 'cos')
            # Calling cos(args, kwargs) (line 329)
            cos_call_result_97181 = invoke(stypy.reporting.localization.Localization(__file__, 329, 25), cos_97173, *[result_mul_97179], **kwargs_97180)
            
            
            # Call to sin(...): (line 329)
            # Processing the call arguments (line 329)
            int_97184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 50), 'int')
            # Getting the type of 'np' (line 329)
            np_97185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 52), 'np', False)
            # Obtaining the member 'pi' of a type (line 329)
            pi_97186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 52), np_97185, 'pi')
            # Applying the binary operator '*' (line 329)
            result_mul_97187 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 50), '*', int_97184, pi_97186)
            
            # Getting the type of 'y' (line 329)
            y_97188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 58), 'y', False)
            # Applying the binary operator '*' (line 329)
            result_mul_97189 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 57), '*', result_mul_97187, y_97188)
            
            # Processing the call keyword arguments (line 329)
            kwargs_97190 = {}
            # Getting the type of 'np' (line 329)
            np_97182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 43), 'np', False)
            # Obtaining the member 'sin' of a type (line 329)
            sin_97183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 43), np_97182, 'sin')
            # Calling sin(args, kwargs) (line 329)
            sin_call_result_97191 = invoke(stypy.reporting.localization.Localization(__file__, 329, 43), sin_97183, *[result_mul_97189], **kwargs_97190)
            
            # Applying the binary operator '*' (line 329)
            result_mul_97192 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 25), '*', cos_call_result_97181, sin_call_result_97191)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', result_mul_97192)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_82' in the type store
            # Getting the type of 'stypy_return_type' (line 329)
            stypy_return_type_97193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_97193)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_82'
            return stypy_return_type_97193

        # Assigning a type to the variable '_stypy_temp_lambda_82' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), '_stypy_temp_lambda_82', _stypy_temp_lambda_82)
        # Getting the type of '_stypy_temp_lambda_82' (line 329)
        _stypy_temp_lambda_82_97194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), '_stypy_temp_lambda_82')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 16), list_97147, _stypy_temp_lambda_82_97194)
        
        # Assigning a type to the variable 'funcs' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'funcs', list_97147)
        
        # Call to seed(...): (line 332)
        # Processing the call arguments (line 332)
        int_97198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'int')
        # Processing the call keyword arguments (line 332)
        kwargs_97199 = {}
        # Getting the type of 'np' (line 332)
        np_97195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 332)
        random_97196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), np_97195, 'random')
        # Obtaining the member 'seed' of a type (line 332)
        seed_97197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), random_97196, 'seed')
        # Calling seed(args, kwargs) (line 332)
        seed_call_result_97200 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), seed_97197, *[int_97198], **kwargs_97199)
        
        
        # Assigning a Subscript to a Name (line 333):
        
        # Assigning a Subscript to a Name (line 333):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_97201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        
        # Call to array(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_97204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_97205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_97206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 32), tuple_97205, int_97206)
        # Adding element type (line 333)
        int_97207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 32), tuple_97205, int_97207)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 30), list_97204, tuple_97205)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_97208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_97209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 39), tuple_97208, int_97209)
        # Adding element type (line 333)
        int_97210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 39), tuple_97208, int_97210)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 30), list_97204, tuple_97208)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_97211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_97212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 46), tuple_97211, int_97212)
        # Adding element type (line 333)
        int_97213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 46), tuple_97211, int_97213)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 30), list_97204, tuple_97211)
        # Adding element type (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_97214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_97215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 53), tuple_97214, int_97215)
        # Adding element type (line 333)
        int_97216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 53), tuple_97214, int_97216)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 30), list_97204, tuple_97214)
        
        # Processing the call keyword arguments (line 333)
        # Getting the type of 'float' (line 333)
        float_97217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 66), 'float', False)
        keyword_97218 = float_97217
        kwargs_97219 = {'dtype': keyword_97218}
        # Getting the type of 'np' (line 333)
        np_97202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 21), 'np', False)
        # Obtaining the member 'array' of a type (line 333)
        array_97203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 21), np_97202, 'array')
        # Calling array(args, kwargs) (line 333)
        array_call_result_97220 = invoke(stypy.reporting.localization.Localization(__file__, 333, 21), array_97203, *[list_97204], **kwargs_97219)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 21), tuple_97201, array_call_result_97220)
        # Adding element type (line 333)
        
        # Call to rand(...): (line 334)
        # Processing the call arguments (line 334)
        int_97224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 36), 'int')
        int_97225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 39), 'int')
        # Applying the binary operator '*' (line 334)
        result_mul_97226 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 36), '*', int_97224, int_97225)
        
        int_97227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 43), 'int')
        # Processing the call keyword arguments (line 334)
        kwargs_97228 = {}
        # Getting the type of 'np' (line 334)
        np_97221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'np', False)
        # Obtaining the member 'random' of a type (line 334)
        random_97222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 21), np_97221, 'random')
        # Obtaining the member 'rand' of a type (line 334)
        rand_97223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 21), random_97222, 'rand')
        # Calling rand(args, kwargs) (line 334)
        rand_call_result_97229 = invoke(stypy.reporting.localization.Localization(__file__, 334, 21), rand_97223, *[result_mul_97226, int_97227], **kwargs_97228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 21), tuple_97201, rand_call_result_97229)
        
        # Getting the type of 'np' (line 333)
        np_97230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 15), 'np')
        # Obtaining the member 'r_' of a type (line 333)
        r__97231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 15), np_97230, 'r_')
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___97232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 15), r__97231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 333)
        subscript_call_result_97233 = invoke(stypy.reporting.localization.Localization(__file__, 333, 15), getitem___97232, tuple_97201)
        
        # Assigning a type to the variable 'grid' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'grid', subscript_call_result_97233)
        
        
        # Call to enumerate(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'funcs' (line 336)
        funcs_97235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 'funcs', False)
        # Processing the call keyword arguments (line 336)
        kwargs_97236 = {}
        # Getting the type of 'enumerate' (line 336)
        enumerate_97234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 336)
        enumerate_call_result_97237 = invoke(stypy.reporting.localization.Localization(__file__, 336, 23), enumerate_97234, *[funcs_97235], **kwargs_97236)
        
        # Testing the type of a for loop iterable (line 336)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 336, 8), enumerate_call_result_97237)
        # Getting the type of the for loop variable (line 336)
        for_loop_var_97238 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 336, 8), enumerate_call_result_97237)
        # Assigning a type to the variable 'j' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_97238))
        # Assigning a type to the variable 'func' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'func', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), for_loop_var_97238))
        # SSA begins for a for statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_accuracy(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'func' (line 337)
        func_97241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'func', False)
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'grid' (line 337)
        grid_97242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 41), 'grid', False)
        keyword_97243 = grid_97242
        float_97244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 51), 'float')
        keyword_97245 = float_97244
        float_97246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 62), 'float')
        keyword_97247 = float_97246
        float_97248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 73), 'float')
        keyword_97249 = float_97248
        str_97250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 41), 'str', 'Function %d')
        # Getting the type of 'j' (line 338)
        j_97251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 57), 'j', False)
        # Applying the binary operator '%' (line 338)
        result_mod_97252 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 41), '%', str_97250, j_97251)
        
        keyword_97253 = result_mod_97252
        kwargs_97254 = {'x': keyword_97243, 'rtol': keyword_97249, 'err_msg': keyword_97253, 'atol': keyword_97247, 'tol': keyword_97245}
        # Getting the type of 'self' (line 337)
        self_97239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 337)
        _check_accuracy_97240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_97239, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 337)
        _check_accuracy_call_result_97255 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), _check_accuracy_97240, *[func_97241], **kwargs_97254)
        
        
        # Call to _check_accuracy(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'func' (line 339)
        func_97258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 33), 'func', False)
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'grid' (line 339)
        grid_97259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 41), 'grid', False)
        keyword_97260 = grid_97259
        float_97261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 51), 'float')
        keyword_97262 = float_97261
        float_97263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 62), 'float')
        keyword_97264 = float_97263
        float_97265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 73), 'float')
        keyword_97266 = float_97265
        str_97267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 41), 'str', 'Function %d')
        # Getting the type of 'j' (line 340)
        j_97268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 57), 'j', False)
        # Applying the binary operator '%' (line 340)
        result_mod_97269 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 41), '%', str_97267, j_97268)
        
        keyword_97270 = result_mod_97269
        # Getting the type of 'True' (line 340)
        True_97271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 68), 'True', False)
        keyword_97272 = True_97271
        kwargs_97273 = {'tol': keyword_97262, 'rescale': keyword_97272, 'err_msg': keyword_97270, 'rtol': keyword_97266, 'atol': keyword_97264, 'x': keyword_97260}
        # Getting the type of 'self' (line 339)
        self_97256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'self', False)
        # Obtaining the member '_check_accuracy' of a type (line 339)
        _check_accuracy_97257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), self_97256, '_check_accuracy')
        # Calling _check_accuracy(args, kwargs) (line 339)
        _check_accuracy_call_result_97274 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), _check_accuracy_97257, *[func_97258], **kwargs_97273)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_97275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97275)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dense'
        return stypy_return_type_97275


    @norecursion
    def test_wrong_ndim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_wrong_ndim'
        module_type_store = module_type_store.open_function_context('test_wrong_ndim', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_wrong_ndim')
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_wrong_ndim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_wrong_ndim', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_wrong_ndim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_wrong_ndim(...)' code ##################

        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to randn(...): (line 343)
        # Processing the call arguments (line 343)
        int_97279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'int')
        int_97280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 32), 'int')
        # Processing the call keyword arguments (line 343)
        kwargs_97281 = {}
        # Getting the type of 'np' (line 343)
        np_97276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 343)
        random_97277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), np_97276, 'random')
        # Obtaining the member 'randn' of a type (line 343)
        randn_97278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), random_97277, 'randn')
        # Calling randn(args, kwargs) (line 343)
        randn_call_result_97282 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), randn_97278, *[int_97279, int_97280], **kwargs_97281)
        
        # Assigning a type to the variable 'x' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'x', randn_call_result_97282)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to randn(...): (line 344)
        # Processing the call arguments (line 344)
        int_97286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 28), 'int')
        # Processing the call keyword arguments (line 344)
        kwargs_97287 = {}
        # Getting the type of 'np' (line 344)
        np_97283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 344)
        random_97284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), np_97283, 'random')
        # Obtaining the member 'randn' of a type (line 344)
        randn_97285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), random_97284, 'randn')
        # Calling randn(args, kwargs) (line 344)
        randn_call_result_97288 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), randn_97285, *[int_97286], **kwargs_97287)
        
        # Assigning a type to the variable 'y' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'y', randn_call_result_97288)
        
        # Call to assert_raises(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'ValueError' (line 345)
        ValueError_97290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 22), 'ValueError', False)
        # Getting the type of 'interpnd' (line 345)
        interpnd_97291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 345)
        CloughTocher2DInterpolator_97292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 34), interpnd_97291, 'CloughTocher2DInterpolator')
        # Getting the type of 'x' (line 345)
        x_97293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 71), 'x', False)
        # Getting the type of 'y' (line 345)
        y_97294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 74), 'y', False)
        # Processing the call keyword arguments (line 345)
        kwargs_97295 = {}
        # Getting the type of 'assert_raises' (line 345)
        assert_raises_97289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 345)
        assert_raises_call_result_97296 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_raises_97289, *[ValueError_97290, CloughTocher2DInterpolator_97292, x_97293, y_97294], **kwargs_97295)
        
        
        # ################# End of 'test_wrong_ndim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_wrong_ndim' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_97297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_wrong_ndim'
        return stypy_return_type_97297


    @norecursion
    def test_pickle(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pickle'
        module_type_store = module_type_store.open_function_context('test_pickle', 347, 4, False)
        # Assigning a type to the variable 'self' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_localization', localization)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_function_name', 'TestCloughTocher2DInterpolator.test_pickle')
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_param_names_list', [])
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCloughTocher2DInterpolator.test_pickle.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.test_pickle', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pickle', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pickle(...)' code ##################

        
        # Call to seed(...): (line 349)
        # Processing the call arguments (line 349)
        int_97301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 23), 'int')
        # Processing the call keyword arguments (line 349)
        kwargs_97302 = {}
        # Getting the type of 'np' (line 349)
        np_97298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 349)
        random_97299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), np_97298, 'random')
        # Obtaining the member 'seed' of a type (line 349)
        seed_97300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), random_97299, 'seed')
        # Calling seed(args, kwargs) (line 349)
        seed_call_result_97303 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), seed_97300, *[int_97301], **kwargs_97302)
        
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to rand(...): (line 350)
        # Processing the call arguments (line 350)
        int_97307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 27), 'int')
        int_97308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 31), 'int')
        # Processing the call keyword arguments (line 350)
        kwargs_97309 = {}
        # Getting the type of 'np' (line 350)
        np_97304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 350)
        random_97305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), np_97304, 'random')
        # Obtaining the member 'rand' of a type (line 350)
        rand_97306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), random_97305, 'rand')
        # Calling rand(args, kwargs) (line 350)
        rand_call_result_97310 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), rand_97306, *[int_97307, int_97308], **kwargs_97309)
        
        # Assigning a type to the variable 'x' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'x', rand_call_result_97310)
        
        # Assigning a BinOp to a Name (line 351):
        
        # Assigning a BinOp to a Name (line 351):
        
        # Call to rand(...): (line 351)
        # Processing the call arguments (line 351)
        int_97314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 27), 'int')
        # Processing the call keyword arguments (line 351)
        kwargs_97315 = {}
        # Getting the type of 'np' (line 351)
        np_97311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 351)
        random_97312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), np_97311, 'random')
        # Obtaining the member 'rand' of a type (line 351)
        rand_97313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), random_97312, 'rand')
        # Calling rand(args, kwargs) (line 351)
        rand_call_result_97316 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), rand_97313, *[int_97314], **kwargs_97315)
        
        complex_97317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 33), 'complex')
        
        # Call to rand(...): (line 351)
        # Processing the call arguments (line 351)
        int_97321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 51), 'int')
        # Processing the call keyword arguments (line 351)
        kwargs_97322 = {}
        # Getting the type of 'np' (line 351)
        np_97318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 36), 'np', False)
        # Obtaining the member 'random' of a type (line 351)
        random_97319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 36), np_97318, 'random')
        # Obtaining the member 'rand' of a type (line 351)
        rand_97320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 36), random_97319, 'rand')
        # Calling rand(args, kwargs) (line 351)
        rand_call_result_97323 = invoke(stypy.reporting.localization.Localization(__file__, 351, 36), rand_97320, *[int_97321], **kwargs_97322)
        
        # Applying the binary operator '*' (line 351)
        result_mul_97324 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 33), '*', complex_97317, rand_call_result_97323)
        
        # Applying the binary operator '+' (line 351)
        result_add_97325 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 12), '+', rand_call_result_97316, result_mul_97324)
        
        # Assigning a type to the variable 'y' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'y', result_add_97325)
        
        # Assigning a Call to a Name (line 353):
        
        # Assigning a Call to a Name (line 353):
        
        # Call to CloughTocher2DInterpolator(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'x' (line 353)
        x_97328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 49), 'x', False)
        # Getting the type of 'y' (line 353)
        y_97329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 52), 'y', False)
        # Processing the call keyword arguments (line 353)
        kwargs_97330 = {}
        # Getting the type of 'interpnd' (line 353)
        interpnd_97326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'interpnd', False)
        # Obtaining the member 'CloughTocher2DInterpolator' of a type (line 353)
        CloughTocher2DInterpolator_97327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 13), interpnd_97326, 'CloughTocher2DInterpolator')
        # Calling CloughTocher2DInterpolator(args, kwargs) (line 353)
        CloughTocher2DInterpolator_call_result_97331 = invoke(stypy.reporting.localization.Localization(__file__, 353, 13), CloughTocher2DInterpolator_97327, *[x_97328, y_97329], **kwargs_97330)
        
        # Assigning a type to the variable 'ip' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'ip', CloughTocher2DInterpolator_call_result_97331)
        
        # Assigning a Call to a Name (line 354):
        
        # Assigning a Call to a Name (line 354):
        
        # Call to loads(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to dumps(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'ip' (line 354)
        ip_97336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 40), 'ip', False)
        # Processing the call keyword arguments (line 354)
        kwargs_97337 = {}
        # Getting the type of 'pickle' (line 354)
        pickle_97334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'pickle', False)
        # Obtaining the member 'dumps' of a type (line 354)
        dumps_97335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 27), pickle_97334, 'dumps')
        # Calling dumps(args, kwargs) (line 354)
        dumps_call_result_97338 = invoke(stypy.reporting.localization.Localization(__file__, 354, 27), dumps_97335, *[ip_97336], **kwargs_97337)
        
        # Processing the call keyword arguments (line 354)
        kwargs_97339 = {}
        # Getting the type of 'pickle' (line 354)
        pickle_97332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 14), 'pickle', False)
        # Obtaining the member 'loads' of a type (line 354)
        loads_97333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 14), pickle_97332, 'loads')
        # Calling loads(args, kwargs) (line 354)
        loads_call_result_97340 = invoke(stypy.reporting.localization.Localization(__file__, 354, 14), loads_97333, *[dumps_call_result_97338], **kwargs_97339)
        
        # Assigning a type to the variable 'ip2' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'ip2', loads_call_result_97340)
        
        # Call to assert_almost_equal(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to ip(...): (line 356)
        # Processing the call arguments (line 356)
        float_97343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 31), 'float')
        float_97344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 36), 'float')
        # Processing the call keyword arguments (line 356)
        kwargs_97345 = {}
        # Getting the type of 'ip' (line 356)
        ip_97342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'ip', False)
        # Calling ip(args, kwargs) (line 356)
        ip_call_result_97346 = invoke(stypy.reporting.localization.Localization(__file__, 356, 28), ip_97342, *[float_97343, float_97344], **kwargs_97345)
        
        
        # Call to ip2(...): (line 356)
        # Processing the call arguments (line 356)
        float_97348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 46), 'float')
        float_97349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 51), 'float')
        # Processing the call keyword arguments (line 356)
        kwargs_97350 = {}
        # Getting the type of 'ip2' (line 356)
        ip2_97347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 42), 'ip2', False)
        # Calling ip2(args, kwargs) (line 356)
        ip2_call_result_97351 = invoke(stypy.reporting.localization.Localization(__file__, 356, 42), ip2_97347, *[float_97348, float_97349], **kwargs_97350)
        
        # Processing the call keyword arguments (line 356)
        kwargs_97352 = {}
        # Getting the type of 'assert_almost_equal' (line 356)
        assert_almost_equal_97341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 356)
        assert_almost_equal_call_result_97353 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), assert_almost_equal_97341, *[ip_call_result_97346, ip2_call_result_97351], **kwargs_97352)
        
        
        # ################# End of 'test_pickle(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pickle' in the type store
        # Getting the type of 'stypy_return_type' (line 347)
        stypy_return_type_97354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_97354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pickle'
        return stypy_return_type_97354


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCloughTocher2DInterpolator.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCloughTocher2DInterpolator' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'TestCloughTocher2DInterpolator', TestCloughTocher2DInterpolator)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
