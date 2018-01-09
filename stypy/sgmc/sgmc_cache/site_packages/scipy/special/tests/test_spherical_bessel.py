
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Tests of spherical Bessel functions.
3: #
4: 
5: import numpy as np
6: from numpy.testing import (assert_almost_equal, assert_allclose,
7:                            assert_array_almost_equal)
8: import pytest
9: from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi
10: 
11: from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn
12: from scipy.integrate import quad
13: 
14: 
15: class TestSphericalJn:
16:     def test_spherical_jn_exact(self):
17:         # http://dlmf.nist.gov/10.49.E3
18:         # Note: exact expression is numerically stable only for small
19:         # n or z >> n.
20:         x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
21:         assert_allclose(spherical_jn(2, x),
22:                         (-1/x + 3/x**3)*sin(x) - 3/x**2*cos(x))
23: 
24:     def test_spherical_jn_recurrence_complex(self):
25:         # http://dlmf.nist.gov/10.51.E1
26:         n = np.array([1, 2, 3, 7, 12])
27:         x = 1.1 + 1.5j
28:         assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1, x),
29:                         (2*n + 1)/x*spherical_jn(n, x))
30: 
31:     def test_spherical_jn_recurrence_real(self):
32:         # http://dlmf.nist.gov/10.51.E1
33:         n = np.array([1, 2, 3, 7, 12])
34:         x = 0.12
35:         assert_allclose(spherical_jn(n - 1, x) + spherical_jn(n + 1,x),
36:                         (2*n + 1)/x*spherical_jn(n, x))
37: 
38:     def test_spherical_jn_inf_real(self):
39:         # http://dlmf.nist.gov/10.52.E3
40:         n = 6
41:         x = np.array([-inf, inf])
42:         assert_allclose(spherical_jn(n, x), np.array([0, 0]))
43: 
44:     def test_spherical_jn_inf_complex(self):
45:         # http://dlmf.nist.gov/10.52.E3
46:         n = 7
47:         x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
48:         assert_allclose(spherical_jn(n, x), np.array([0, 0, inf*(1+1j)]))
49: 
50:     def test_spherical_jn_large_arg_1(self):
51:         # https://github.com/scipy/scipy/issues/2165
52:         # Reference value computed using mpmath, via
53:         # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
54:         assert_allclose(spherical_jn(2, 3350.507), -0.00029846226538040747)
55: 
56:     def test_spherical_jn_large_arg_2(self):
57:         # https://github.com/scipy/scipy/issues/1641
58:         # Reference value computed using mpmath, via
59:         # besselj(n + mpf(1)/2, z)*sqrt(pi/(2*z))
60:         assert_allclose(spherical_jn(2, 10000), 3.0590002633029811e-05)
61: 
62:     def test_spherical_jn_at_zero(self):
63:         # http://dlmf.nist.gov/10.52.E1
64:         # But note that n = 0 is a special case: j0 = sin(x)/x -> 1
65:         n = np.array([0, 1, 2, 5, 10, 100])
66:         x = 0
67:         assert_allclose(spherical_jn(n, x), np.array([1, 0, 0, 0, 0, 0]))
68: 
69: 
70: class TestSphericalYn:
71:     def test_spherical_yn_exact(self):
72:         # http://dlmf.nist.gov/10.49.E5
73:         # Note: exact expression is numerically stable only for small
74:         # n or z >> n.
75:         x = np.array([0.12, 1.23, 12.34, 123.45, 1234.5])
76:         assert_allclose(spherical_yn(2, x),
77:                         (1/x - 3/x**3)*cos(x) - 3/x**2*sin(x))
78: 
79:     def test_spherical_yn_recurrence_real(self):
80:         # http://dlmf.nist.gov/10.51.E1
81:         n = np.array([1, 2, 3, 7, 12])
82:         x = 0.12
83:         assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1,x),
84:                         (2*n + 1)/x*spherical_yn(n, x))
85: 
86:     def test_spherical_yn_recurrence_complex(self):
87:         # http://dlmf.nist.gov/10.51.E1
88:         n = np.array([1, 2, 3, 7, 12])
89:         x = 1.1 + 1.5j
90:         assert_allclose(spherical_yn(n - 1, x) + spherical_yn(n + 1, x),
91:                         (2*n + 1)/x*spherical_yn(n, x))
92: 
93:     def test_spherical_yn_inf_real(self):
94:         # http://dlmf.nist.gov/10.52.E3
95:         n = 6
96:         x = np.array([-inf, inf])
97:         assert_allclose(spherical_yn(n, x), np.array([0, 0]))
98: 
99:     def test_spherical_yn_inf_complex(self):
100:         # http://dlmf.nist.gov/10.52.E3
101:         n = 7
102:         x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
103:         assert_allclose(spherical_yn(n, x), np.array([0, 0, inf*(1+1j)]))
104: 
105:     def test_spherical_yn_at_zero(self):
106:         # http://dlmf.nist.gov/10.52.E2
107:         n = np.array([0, 1, 2, 5, 10, 100])
108:         x = 0
109:         assert_allclose(spherical_yn(n, x), -inf*np.ones(shape=n.shape))
110: 
111:     def test_spherical_yn_at_zero_complex(self):
112:         # Consistently with numpy:
113:         # >>> -np.cos(0)/0
114:         # -inf
115:         # >>> -np.cos(0+0j)/(0+0j)
116:         # (-inf + nan*j)
117:         n = np.array([0, 1, 2, 5, 10, 100])
118:         x = 0 + 0j
119:         assert_allclose(spherical_yn(n, x), nan*np.ones(shape=n.shape))
120: 
121: 
122: class TestSphericalJnYnCrossProduct:
123:     def test_spherical_jn_yn_cross_product_1(self):
124:         # http://dlmf.nist.gov/10.50.E3
125:         n = np.array([1, 5, 8])
126:         x = np.array([0.1, 1, 10])
127:         left = (spherical_jn(n + 1, x) * spherical_yn(n, x) -
128:                 spherical_jn(n, x) * spherical_yn(n + 1, x))
129:         right = 1/x**2
130:         assert_allclose(left, right)
131: 
132:     def test_spherical_jn_yn_cross_product_2(self):
133:         # http://dlmf.nist.gov/10.50.E3
134:         n = np.array([1, 5, 8])
135:         x = np.array([0.1, 1, 10])
136:         left = (spherical_jn(n + 2, x) * spherical_yn(n, x) -
137:                 spherical_jn(n, x) * spherical_yn(n + 2, x))
138:         right = (2*n + 3)/x**3
139:         assert_allclose(left, right)
140: 
141: 
142: class TestSphericalIn:
143:     def test_spherical_in_exact(self):
144:         # http://dlmf.nist.gov/10.49.E9
145:         x = np.array([0.12, 1.23, 12.34, 123.45])
146:         assert_allclose(spherical_in(2, x),
147:                         (1/x + 3/x**3)*sinh(x) - 3/x**2*cosh(x))
148: 
149:     def test_spherical_in_recurrence_real(self):
150:         # http://dlmf.nist.gov/10.51.E4
151:         n = np.array([1, 2, 3, 7, 12])
152:         x = 0.12
153:         assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
154:                         (2*n + 1)/x*spherical_in(n, x))
155: 
156:     def test_spherical_in_recurrence_complex(self):
157:         # http://dlmf.nist.gov/10.51.E1
158:         n = np.array([1, 2, 3, 7, 12])
159:         x = 1.1 + 1.5j
160:         assert_allclose(spherical_in(n - 1, x) - spherical_in(n + 1,x),
161:                         (2*n + 1)/x*spherical_in(n, x))
162: 
163:     def test_spherical_in_inf_real(self):
164:         # http://dlmf.nist.gov/10.52.E3
165:         n = 5
166:         x = np.array([-inf, inf])
167:         assert_allclose(spherical_in(n, x), np.array([-inf, inf]))
168: 
169:     def test_spherical_in_inf_complex(self):
170:         # http://dlmf.nist.gov/10.52.E5
171:         # Ideally, i1n(n, 1j*inf) = 0 and i1n(n, (1+1j)*inf) = (1+1j)*inf, but
172:         # this appears impossible to achieve because C99 regards any complex
173:         # value with at least one infinite  part as a complex infinity, so
174:         # 1j*inf cannot be distinguished from (1+1j)*inf.  Therefore, nan is
175:         # the correct return value.
176:         n = 7
177:         x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
178:         assert_allclose(spherical_in(n, x), np.array([-inf, inf, nan]))
179: 
180:     def test_spherical_in_at_zero(self):
181:         # http://dlmf.nist.gov/10.52.E1
182:         # But note that n = 0 is a special case: i0 = sinh(x)/x -> 1
183:         n = np.array([0, 1, 2, 5, 10, 100])
184:         x = 0
185:         assert_allclose(spherical_in(n, x), np.array([1, 0, 0, 0, 0, 0]))
186: 
187: 
188: class TestSphericalKn:
189:     def test_spherical_kn_exact(self):
190:         # http://dlmf.nist.gov/10.49.E13
191:         x = np.array([0.12, 1.23, 12.34, 123.45])
192:         assert_allclose(spherical_kn(2, x),
193:                         pi/2*exp(-x)*(1/x + 3/x**2 + 3/x**3))
194: 
195:     def test_spherical_kn_recurrence_real(self):
196:         # http://dlmf.nist.gov/10.51.E4
197:         n = np.array([1, 2, 3, 7, 12])
198:         x = 0.12
199:         assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
200:                         (-1)**n*(2*n + 1)/x*spherical_kn(n, x))
201: 
202:     def test_spherical_kn_recurrence_complex(self):
203:         # http://dlmf.nist.gov/10.51.E4
204:         n = np.array([1, 2, 3, 7, 12])
205:         x = 1.1 + 1.5j
206:         assert_allclose((-1)**(n - 1)*spherical_kn(n - 1, x) - (-1)**(n + 1)*spherical_kn(n + 1,x),
207:                         (-1)**n*(2*n + 1)/x*spherical_kn(n, x))
208: 
209:     def test_spherical_kn_inf_real(self):
210:         # http://dlmf.nist.gov/10.52.E6
211:         n = 5
212:         x = np.array([-inf, inf])
213:         assert_allclose(spherical_kn(n, x), np.array([-inf, 0]))
214: 
215:     def test_spherical_kn_inf_complex(self):
216:         # http://dlmf.nist.gov/10.52.E6
217:         # The behavior at complex infinity depends on the sign of the real
218:         # part: if Re(z) >= 0, then the limit is 0; if Re(z) < 0, then it's
219:         # z*inf.  This distinction cannot be captured, so we return nan.
220:         n = 7
221:         x = np.array([-inf + 0j, inf + 0j, inf*(1+1j)])
222:         assert_allclose(spherical_kn(n, x), np.array([-inf, 0, nan]))
223: 
224:     def test_spherical_kn_at_zero(self):
225:         # http://dlmf.nist.gov/10.52.E2
226:         n = np.array([0, 1, 2, 5, 10, 100])
227:         x = 0
228:         assert_allclose(spherical_kn(n, x), inf*np.ones(shape=n.shape))
229: 
230:     def test_spherical_kn_at_zero_complex(self):
231:         # http://dlmf.nist.gov/10.52.E2
232:         n = np.array([0, 1, 2, 5, 10, 100])
233:         x = 0 + 0j
234:         assert_allclose(spherical_kn(n, x), nan*np.ones(shape=n.shape))
235: 
236: 
237: class SphericalDerivativesTestCase:
238:     def fundamental_theorem(self, n, a, b):
239:         integral, tolerance = quad(lambda z: self.df(n, z), a, b)
240:         assert_allclose(integral,
241:                         self.f(n, b) - self.f(n, a),
242:                         atol=tolerance)
243: 
244:     @pytest.mark.slow
245:     def test_fundamental_theorem_0(self):
246:         self.fundamental_theorem(0, 3.0, 15.0)
247: 
248:     @pytest.mark.slow
249:     def test_fundamental_theorem_7(self):
250:         self.fundamental_theorem(7, 0.5, 1.2)
251: 
252: 
253: class TestSphericalJnDerivatives(SphericalDerivativesTestCase):
254:     def f(self, n, z):
255:         return spherical_jn(n, z)
256: 
257:     def df(self, n, z):
258:         return spherical_jn(n, z, derivative=True)
259: 
260:     def test_spherical_jn_d_zero(self):
261:         n = np.array([1, 2, 3, 7, 15])
262:         assert_allclose(spherical_jn(n, 0, derivative=True),
263:                         np.zeros(5))
264: 
265: 
266: class TestSphericalYnDerivatives(SphericalDerivativesTestCase):
267:     def f(self, n, z):
268:         return spherical_yn(n, z)
269: 
270:     def df(self, n, z):
271:         return spherical_yn(n, z, derivative=True)
272: 
273: 
274: class TestSphericalInDerivatives(SphericalDerivativesTestCase):
275:     def f(self, n, z):
276:         return spherical_in(n, z)
277: 
278:     def df(self, n, z):
279:         return spherical_in(n, z, derivative=True)
280: 
281:     def test_spherical_in_d_zero(self):
282:         n = np.array([1, 2, 3, 7, 15])
283:         assert_allclose(spherical_in(n, 0, derivative=True),
284:                         np.zeros(5))
285: 
286: 
287: class TestSphericalKnDerivatives(SphericalDerivativesTestCase):
288:     def f(self, n, z):
289:         return spherical_kn(n, z)
290: 
291:     def df(self, n, z):
292:         return spherical_kn(n, z, derivative=True)
293: 
294: 
295: class TestSphericalOld:
296:     # These are tests from the TestSpherical class of test_basic.py,
297:     # rewritten to use spherical_* instead of sph_* but otherwise unchanged.
298: 
299:     def test_sph_in(self):
300:         # This test reproduces test_basic.TestSpherical.test_sph_in.
301:         i1n = np.empty((2,2))
302:         x = 0.2
303: 
304:         i1n[0][0] = spherical_in(0, x)
305:         i1n[0][1] = spherical_in(1, x)
306:         i1n[1][0] = spherical_in(0, x, derivative=True)
307:         i1n[1][1] = spherical_in(1, x, derivative=True)
308: 
309:         inp0 = (i1n[0][1])
310:         inp1 = (i1n[0][0] - 2.0/0.2 * i1n[0][1])
311:         assert_array_almost_equal(i1n[0],np.array([1.0066800127054699381,
312:                                                 0.066933714568029540839]),12)
313:         assert_array_almost_equal(i1n[1],[inp0,inp1],12)
314: 
315:     def test_sph_in_kn_order0(self):
316:         x = 1.
317:         sph_i0 = np.empty((2,))
318:         sph_i0[0] = spherical_in(0, x)
319:         sph_i0[1] = spherical_in(0, x, derivative=True)
320:         sph_i0_expected = np.array([np.sinh(x)/x,
321:                                     np.cosh(x)/x-np.sinh(x)/x**2])
322:         assert_array_almost_equal(r_[sph_i0], sph_i0_expected)
323: 
324:         sph_k0 = np.empty((2,))
325:         sph_k0[0] = spherical_kn(0, x)
326:         sph_k0[1] = spherical_kn(0, x, derivative=True)
327:         sph_k0_expected = np.array([0.5*pi*exp(-x)/x,
328:                                     -0.5*pi*exp(-x)*(1/x+1/x**2)])
329:         assert_array_almost_equal(r_[sph_k0], sph_k0_expected)
330: 
331:     def test_sph_jn(self):
332:         s1 = np.empty((2,3))
333:         x = 0.2
334: 
335:         s1[0][0] = spherical_jn(0, x)
336:         s1[0][1] = spherical_jn(1, x)
337:         s1[0][2] = spherical_jn(2, x)
338:         s1[1][0] = spherical_jn(0, x, derivative=True)
339:         s1[1][1] = spherical_jn(1, x, derivative=True)
340:         s1[1][2] = spherical_jn(2, x, derivative=True)
341: 
342:         s10 = -s1[0][1]
343:         s11 = s1[0][0]-2.0/0.2*s1[0][1]
344:         s12 = s1[0][1]-3.0/0.2*s1[0][2]
345:         assert_array_almost_equal(s1[0],[0.99334665397530607731,
346:                                       0.066400380670322230863,
347:                                       0.0026590560795273856680],12)
348:         assert_array_almost_equal(s1[1],[s10,s11,s12],12)
349: 
350:     def test_sph_kn(self):
351:         kn = np.empty((2,3))
352:         x = 0.2
353: 
354:         kn[0][0] = spherical_kn(0, x)
355:         kn[0][1] = spherical_kn(1, x)
356:         kn[0][2] = spherical_kn(2, x)
357:         kn[1][0] = spherical_kn(0, x, derivative=True)
358:         kn[1][1] = spherical_kn(1, x, derivative=True)
359:         kn[1][2] = spherical_kn(2, x, derivative=True)
360: 
361:         kn0 = -kn[0][1]
362:         kn1 = -kn[0][0]-2.0/0.2*kn[0][1]
363:         kn2 = -kn[0][1]-3.0/0.2*kn[0][2]
364:         assert_array_almost_equal(kn[0],[6.4302962978445670140,
365:                                          38.581777787067402086,
366:                                          585.15696310385559829],12)
367:         assert_array_almost_equal(kn[1],[kn0,kn1,kn2],9)
368: 
369:     def test_sph_yn(self):
370:         sy1 = spherical_yn(2, 0.2)
371:         sy2 = spherical_yn(0, 0.2)
372:         assert_almost_equal(sy1,-377.52483,5)  # previous values in the system
373:         assert_almost_equal(sy2,-4.9003329,5)
374:         sphpy = (spherical_yn(0, 0.2) - 2*spherical_yn(2, 0.2))/3
375:         sy3 = spherical_yn(1, 0.2, derivative=True)
376:         assert_almost_equal(sy3,sphpy,4)  # compare correct derivative val. (correct =-system val).
377: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_560890) is not StypyTypeError):

    if (import_560890 != 'pyd_module'):
        __import__(import_560890)
        sys_modules_560891 = sys.modules[import_560890]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_560891.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_560890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_almost_equal, assert_allclose, assert_array_almost_equal' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_560892) is not StypyTypeError):

    if (import_560892 != 'pyd_module'):
        __import__(import_560892)
        sys_modules_560893 = sys.modules[import_560892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_560893.module_type_store, module_type_store, ['assert_almost_equal', 'assert_allclose', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_560893, sys_modules_560893.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_allclose, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_allclose', 'assert_array_almost_equal'], [assert_almost_equal, assert_allclose, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_560892)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_560894) is not StypyTypeError):

    if (import_560894 != 'pyd_module'):
        __import__(import_560894)
        sys_modules_560895 = sys.modules[import_560894]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_560895.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_560894)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560896 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_560896) is not StypyTypeError):

    if (import_560896 != 'pyd_module'):
        __import__(import_560896)
        sys_modules_560897 = sys.modules[import_560896]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', sys_modules_560897.module_type_store, module_type_store, ['sin', 'cos', 'sinh', 'cosh', 'exp', 'inf', 'nan', 'r_', 'pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_560897, sys_modules_560897.module_type_store, module_type_store)
    else:
        from numpy import sin, cos, sinh, cosh, exp, inf, nan, r_, pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', None, module_type_store, ['sin', 'cos', 'sinh', 'cosh', 'exp', 'inf', 'nan', 'r_', 'pi'], [sin, cos, sinh, cosh, exp, inf, nan, r_, pi])

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_560896)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560898 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special')

if (type(import_560898) is not StypyTypeError):

    if (import_560898 != 'pyd_module'):
        __import__(import_560898)
        sys_modules_560899 = sys.modules[import_560898]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', sys_modules_560899.module_type_store, module_type_store, ['spherical_jn', 'spherical_yn', 'spherical_in', 'spherical_kn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_560899, sys_modules_560899.module_type_store, module_type_store)
    else:
        from scipy.special import spherical_jn, spherical_yn, spherical_in, spherical_kn

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', None, module_type_store, ['spherical_jn', 'spherical_yn', 'spherical_in', 'spherical_kn'], [spherical_jn, spherical_yn, spherical_in, spherical_kn])

else:
    # Assigning a type to the variable 'scipy.special' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special', import_560898)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.integrate import quad' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate')

if (type(import_560900) is not StypyTypeError):

    if (import_560900 != 'pyd_module'):
        __import__(import_560900)
        sys_modules_560901 = sys.modules[import_560900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', sys_modules_560901.module_type_store, module_type_store, ['quad'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_560901, sys_modules_560901.module_type_store, module_type_store)
    else:
        from scipy.integrate import quad

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', None, module_type_store, ['quad'], [quad])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.integrate', import_560900)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

# Declaration of the 'TestSphericalJn' class

class TestSphericalJn:

    @norecursion
    def test_spherical_jn_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_exact'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_exact', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_exact')
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_exact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_exact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_exact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_exact(...)' code ##################

        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to array(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_560904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        float_560905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_560904, float_560905)
        # Adding element type (line 20)
        float_560906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_560904, float_560906)
        # Adding element type (line 20)
        float_560907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_560904, float_560907)
        # Adding element type (line 20)
        float_560908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_560904, float_560908)
        # Adding element type (line 20)
        float_560909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 21), list_560904, float_560909)
        
        # Processing the call keyword arguments (line 20)
        kwargs_560910 = {}
        # Getting the type of 'np' (line 20)
        np_560902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 20)
        array_560903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), np_560902, 'array')
        # Calling array(args, kwargs) (line 20)
        array_call_result_560911 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), array_560903, *[list_560904], **kwargs_560910)
        
        # Assigning a type to the variable 'x' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'x', array_call_result_560911)
        
        # Call to assert_allclose(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Call to spherical_jn(...): (line 21)
        # Processing the call arguments (line 21)
        int_560914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'int')
        # Getting the type of 'x' (line 21)
        x_560915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 40), 'x', False)
        # Processing the call keyword arguments (line 21)
        kwargs_560916 = {}
        # Getting the type of 'spherical_jn' (line 21)
        spherical_jn_560913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 21)
        spherical_jn_call_result_560917 = invoke(stypy.reporting.localization.Localization(__file__, 21, 24), spherical_jn_560913, *[int_560914, x_560915], **kwargs_560916)
        
        int_560918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')
        # Getting the type of 'x' (line 22)
        x_560919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'x', False)
        # Applying the binary operator 'div' (line 22)
        result_div_560920 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 25), 'div', int_560918, x_560919)
        
        int_560921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
        # Getting the type of 'x' (line 22)
        x_560922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), 'x', False)
        int_560923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 37), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_560924 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 34), '**', x_560922, int_560923)
        
        # Applying the binary operator 'div' (line 22)
        result_div_560925 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 32), 'div', int_560921, result_pow_560924)
        
        # Applying the binary operator '+' (line 22)
        result_add_560926 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 25), '+', result_div_560920, result_div_560925)
        
        
        # Call to sin(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'x' (line 22)
        x_560928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 44), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_560929 = {}
        # Getting the type of 'sin' (line 22)
        sin_560927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 40), 'sin', False)
        # Calling sin(args, kwargs) (line 22)
        sin_call_result_560930 = invoke(stypy.reporting.localization.Localization(__file__, 22, 40), sin_560927, *[x_560928], **kwargs_560929)
        
        # Applying the binary operator '*' (line 22)
        result_mul_560931 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 24), '*', result_add_560926, sin_call_result_560930)
        
        int_560932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 49), 'int')
        # Getting the type of 'x' (line 22)
        x_560933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'x', False)
        int_560934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 54), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_560935 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 51), '**', x_560933, int_560934)
        
        # Applying the binary operator 'div' (line 22)
        result_div_560936 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 49), 'div', int_560932, result_pow_560935)
        
        
        # Call to cos(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'x' (line 22)
        x_560938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 60), 'x', False)
        # Processing the call keyword arguments (line 22)
        kwargs_560939 = {}
        # Getting the type of 'cos' (line 22)
        cos_560937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 56), 'cos', False)
        # Calling cos(args, kwargs) (line 22)
        cos_call_result_560940 = invoke(stypy.reporting.localization.Localization(__file__, 22, 56), cos_560937, *[x_560938], **kwargs_560939)
        
        # Applying the binary operator '*' (line 22)
        result_mul_560941 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 55), '*', result_div_560936, cos_call_result_560940)
        
        # Applying the binary operator '-' (line 22)
        result_sub_560942 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 24), '-', result_mul_560931, result_mul_560941)
        
        # Processing the call keyword arguments (line 21)
        kwargs_560943 = {}
        # Getting the type of 'assert_allclose' (line 21)
        assert_allclose_560912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 21)
        assert_allclose_call_result_560944 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), assert_allclose_560912, *[spherical_jn_call_result_560917, result_sub_560942], **kwargs_560943)
        
        
        # ################# End of 'test_spherical_jn_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_560945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_exact'
        return stypy_return_type_560945


    @norecursion
    def test_spherical_jn_recurrence_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_recurrence_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_recurrence_complex', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_recurrence_complex')
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_recurrence_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_recurrence_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_recurrence_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_recurrence_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 26):
        
        # Assigning a Call to a Name (line 26):
        
        # Call to array(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_560948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        int_560949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_560948, int_560949)
        # Adding element type (line 26)
        int_560950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_560948, int_560950)
        # Adding element type (line 26)
        int_560951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_560948, int_560951)
        # Adding element type (line 26)
        int_560952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_560948, int_560952)
        # Adding element type (line 26)
        int_560953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), list_560948, int_560953)
        
        # Processing the call keyword arguments (line 26)
        kwargs_560954 = {}
        # Getting the type of 'np' (line 26)
        np_560946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 26)
        array_560947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), np_560946, 'array')
        # Calling array(args, kwargs) (line 26)
        array_call_result_560955 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), array_560947, *[list_560948], **kwargs_560954)
        
        # Assigning a type to the variable 'n' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'n', array_call_result_560955)
        
        # Assigning a BinOp to a Name (line 27):
        
        # Assigning a BinOp to a Name (line 27):
        float_560956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 12), 'float')
        complex_560957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'complex')
        # Applying the binary operator '+' (line 27)
        result_add_560958 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 12), '+', float_560956, complex_560957)
        
        # Assigning a type to the variable 'x' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', result_add_560958)
        
        # Call to assert_allclose(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to spherical_jn(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'n' (line 28)
        n_560961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 37), 'n', False)
        int_560962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'int')
        # Applying the binary operator '-' (line 28)
        result_sub_560963 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 37), '-', n_560961, int_560962)
        
        # Getting the type of 'x' (line 28)
        x_560964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_560965 = {}
        # Getting the type of 'spherical_jn' (line 28)
        spherical_jn_560960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 28)
        spherical_jn_call_result_560966 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), spherical_jn_560960, *[result_sub_560963, x_560964], **kwargs_560965)
        
        
        # Call to spherical_jn(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'n' (line 28)
        n_560968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 62), 'n', False)
        int_560969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 66), 'int')
        # Applying the binary operator '+' (line 28)
        result_add_560970 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 62), '+', n_560968, int_560969)
        
        # Getting the type of 'x' (line 28)
        x_560971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 69), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_560972 = {}
        # Getting the type of 'spherical_jn' (line 28)
        spherical_jn_560967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 28)
        spherical_jn_call_result_560973 = invoke(stypy.reporting.localization.Localization(__file__, 28, 49), spherical_jn_560967, *[result_add_560970, x_560971], **kwargs_560972)
        
        # Applying the binary operator '+' (line 28)
        result_add_560974 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 24), '+', spherical_jn_call_result_560966, spherical_jn_call_result_560973)
        
        int_560975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'int')
        # Getting the type of 'n' (line 29)
        n_560976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'n', False)
        # Applying the binary operator '*' (line 29)
        result_mul_560977 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 25), '*', int_560975, n_560976)
        
        int_560978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 31), 'int')
        # Applying the binary operator '+' (line 29)
        result_add_560979 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 25), '+', result_mul_560977, int_560978)
        
        # Getting the type of 'x' (line 29)
        x_560980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'x', False)
        # Applying the binary operator 'div' (line 29)
        result_div_560981 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 24), 'div', result_add_560979, x_560980)
        
        
        # Call to spherical_jn(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'n' (line 29)
        n_560983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 49), 'n', False)
        # Getting the type of 'x' (line 29)
        x_560984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 52), 'x', False)
        # Processing the call keyword arguments (line 29)
        kwargs_560985 = {}
        # Getting the type of 'spherical_jn' (line 29)
        spherical_jn_560982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 29)
        spherical_jn_call_result_560986 = invoke(stypy.reporting.localization.Localization(__file__, 29, 36), spherical_jn_560982, *[n_560983, x_560984], **kwargs_560985)
        
        # Applying the binary operator '*' (line 29)
        result_mul_560987 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 35), '*', result_div_560981, spherical_jn_call_result_560986)
        
        # Processing the call keyword arguments (line 28)
        kwargs_560988 = {}
        # Getting the type of 'assert_allclose' (line 28)
        assert_allclose_560959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 28)
        assert_allclose_call_result_560989 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert_allclose_560959, *[result_add_560974, result_mul_560987], **kwargs_560988)
        
        
        # ################# End of 'test_spherical_jn_recurrence_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_recurrence_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_560990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_560990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_recurrence_complex'
        return stypy_return_type_560990


    @norecursion
    def test_spherical_jn_recurrence_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_recurrence_real'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_recurrence_real', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_recurrence_real')
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_recurrence_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_recurrence_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_recurrence_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_recurrence_real(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to array(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_560993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        # Adding element type (line 33)
        int_560994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_560993, int_560994)
        # Adding element type (line 33)
        int_560995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_560993, int_560995)
        # Adding element type (line 33)
        int_560996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_560993, int_560996)
        # Adding element type (line 33)
        int_560997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_560993, int_560997)
        # Adding element type (line 33)
        int_560998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), list_560993, int_560998)
        
        # Processing the call keyword arguments (line 33)
        kwargs_560999 = {}
        # Getting the type of 'np' (line 33)
        np_560991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 33)
        array_560992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), np_560991, 'array')
        # Calling array(args, kwargs) (line 33)
        array_call_result_561000 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), array_560992, *[list_560993], **kwargs_560999)
        
        # Assigning a type to the variable 'n' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'n', array_call_result_561000)
        
        # Assigning a Num to a Name (line 34):
        
        # Assigning a Num to a Name (line 34):
        float_561001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'float')
        # Assigning a type to the variable 'x' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'x', float_561001)
        
        # Call to assert_allclose(...): (line 35)
        # Processing the call arguments (line 35)
        
        # Call to spherical_jn(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'n' (line 35)
        n_561004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'n', False)
        int_561005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 41), 'int')
        # Applying the binary operator '-' (line 35)
        result_sub_561006 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 37), '-', n_561004, int_561005)
        
        # Getting the type of 'x' (line 35)
        x_561007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 44), 'x', False)
        # Processing the call keyword arguments (line 35)
        kwargs_561008 = {}
        # Getting the type of 'spherical_jn' (line 35)
        spherical_jn_561003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 35)
        spherical_jn_call_result_561009 = invoke(stypy.reporting.localization.Localization(__file__, 35, 24), spherical_jn_561003, *[result_sub_561006, x_561007], **kwargs_561008)
        
        
        # Call to spherical_jn(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'n' (line 35)
        n_561011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 62), 'n', False)
        int_561012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 66), 'int')
        # Applying the binary operator '+' (line 35)
        result_add_561013 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 62), '+', n_561011, int_561012)
        
        # Getting the type of 'x' (line 35)
        x_561014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 68), 'x', False)
        # Processing the call keyword arguments (line 35)
        kwargs_561015 = {}
        # Getting the type of 'spherical_jn' (line 35)
        spherical_jn_561010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 35)
        spherical_jn_call_result_561016 = invoke(stypy.reporting.localization.Localization(__file__, 35, 49), spherical_jn_561010, *[result_add_561013, x_561014], **kwargs_561015)
        
        # Applying the binary operator '+' (line 35)
        result_add_561017 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 24), '+', spherical_jn_call_result_561009, spherical_jn_call_result_561016)
        
        int_561018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'int')
        # Getting the type of 'n' (line 36)
        n_561019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'n', False)
        # Applying the binary operator '*' (line 36)
        result_mul_561020 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '*', int_561018, n_561019)
        
        int_561021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 31), 'int')
        # Applying the binary operator '+' (line 36)
        result_add_561022 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '+', result_mul_561020, int_561021)
        
        # Getting the type of 'x' (line 36)
        x_561023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'x', False)
        # Applying the binary operator 'div' (line 36)
        result_div_561024 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), 'div', result_add_561022, x_561023)
        
        
        # Call to spherical_jn(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'n' (line 36)
        n_561026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 49), 'n', False)
        # Getting the type of 'x' (line 36)
        x_561027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 52), 'x', False)
        # Processing the call keyword arguments (line 36)
        kwargs_561028 = {}
        # Getting the type of 'spherical_jn' (line 36)
        spherical_jn_561025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 36)
        spherical_jn_call_result_561029 = invoke(stypy.reporting.localization.Localization(__file__, 36, 36), spherical_jn_561025, *[n_561026, x_561027], **kwargs_561028)
        
        # Applying the binary operator '*' (line 36)
        result_mul_561030 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 35), '*', result_div_561024, spherical_jn_call_result_561029)
        
        # Processing the call keyword arguments (line 35)
        kwargs_561031 = {}
        # Getting the type of 'assert_allclose' (line 35)
        assert_allclose_561002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 35)
        assert_allclose_call_result_561032 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_allclose_561002, *[result_add_561017, result_mul_561030], **kwargs_561031)
        
        
        # ################# End of 'test_spherical_jn_recurrence_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_recurrence_real' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_561033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_recurrence_real'
        return stypy_return_type_561033


    @norecursion
    def test_spherical_jn_inf_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_inf_real'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_inf_real', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_inf_real')
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_inf_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_inf_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_inf_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_inf_real(...)' code ##################

        
        # Assigning a Num to a Name (line 40):
        
        # Assigning a Num to a Name (line 40):
        int_561034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'int')
        # Assigning a type to the variable 'n' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'n', int_561034)
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to array(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_561037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        # Adding element type (line 41)
        
        # Getting the type of 'inf' (line 41)
        inf_561038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 41)
        result___neg___561039 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 22), 'usub', inf_561038)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), list_561037, result___neg___561039)
        # Adding element type (line 41)
        # Getting the type of 'inf' (line 41)
        inf_561040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), list_561037, inf_561040)
        
        # Processing the call keyword arguments (line 41)
        kwargs_561041 = {}
        # Getting the type of 'np' (line 41)
        np_561035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 41)
        array_561036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), np_561035, 'array')
        # Calling array(args, kwargs) (line 41)
        array_call_result_561042 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), array_561036, *[list_561037], **kwargs_561041)
        
        # Assigning a type to the variable 'x' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'x', array_call_result_561042)
        
        # Call to assert_allclose(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to spherical_jn(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'n' (line 42)
        n_561045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'n', False)
        # Getting the type of 'x' (line 42)
        x_561046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'x', False)
        # Processing the call keyword arguments (line 42)
        kwargs_561047 = {}
        # Getting the type of 'spherical_jn' (line 42)
        spherical_jn_561044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 42)
        spherical_jn_call_result_561048 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), spherical_jn_561044, *[n_561045, x_561046], **kwargs_561047)
        
        
        # Call to array(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_561051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        # Adding element type (line 42)
        int_561052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 53), list_561051, int_561052)
        # Adding element type (line 42)
        int_561053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 53), list_561051, int_561053)
        
        # Processing the call keyword arguments (line 42)
        kwargs_561054 = {}
        # Getting the type of 'np' (line 42)
        np_561049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 42)
        array_561050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 44), np_561049, 'array')
        # Calling array(args, kwargs) (line 42)
        array_call_result_561055 = invoke(stypy.reporting.localization.Localization(__file__, 42, 44), array_561050, *[list_561051], **kwargs_561054)
        
        # Processing the call keyword arguments (line 42)
        kwargs_561056 = {}
        # Getting the type of 'assert_allclose' (line 42)
        assert_allclose_561043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 42)
        assert_allclose_call_result_561057 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_allclose_561043, *[spherical_jn_call_result_561048, array_call_result_561055], **kwargs_561056)
        
        
        # ################# End of 'test_spherical_jn_inf_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_inf_real' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_561058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561058)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_inf_real'
        return stypy_return_type_561058


    @norecursion
    def test_spherical_jn_inf_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_inf_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_inf_complex', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_inf_complex')
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_inf_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_inf_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_inf_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_inf_complex(...)' code ##################

        
        # Assigning a Num to a Name (line 46):
        
        # Assigning a Num to a Name (line 46):
        int_561059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'int')
        # Assigning a type to the variable 'n' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'n', int_561059)
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to array(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_561062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        
        # Getting the type of 'inf' (line 47)
        inf_561063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 47)
        result___neg___561064 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 22), 'usub', inf_561063)
        
        complex_561065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'complex')
        # Applying the binary operator '+' (line 47)
        result_add_561066 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 22), '+', result___neg___561064, complex_561065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_561062, result_add_561066)
        # Adding element type (line 47)
        # Getting the type of 'inf' (line 47)
        inf_561067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'inf', False)
        complex_561068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 39), 'complex')
        # Applying the binary operator '+' (line 47)
        result_add_561069 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 33), '+', inf_561067, complex_561068)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_561062, result_add_561069)
        # Adding element type (line 47)
        # Getting the type of 'inf' (line 47)
        inf_561070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 43), 'inf', False)
        int_561071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'int')
        complex_561072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'complex')
        # Applying the binary operator '+' (line 47)
        result_add_561073 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 48), '+', int_561071, complex_561072)
        
        # Applying the binary operator '*' (line 47)
        result_mul_561074 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 43), '*', inf_561070, result_add_561073)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 21), list_561062, result_mul_561074)
        
        # Processing the call keyword arguments (line 47)
        kwargs_561075 = {}
        # Getting the type of 'np' (line 47)
        np_561060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 47)
        array_561061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), np_561060, 'array')
        # Calling array(args, kwargs) (line 47)
        array_call_result_561076 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), array_561061, *[list_561062], **kwargs_561075)
        
        # Assigning a type to the variable 'x' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'x', array_call_result_561076)
        
        # Call to assert_allclose(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to spherical_jn(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'n' (line 48)
        n_561079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'n', False)
        # Getting the type of 'x' (line 48)
        x_561080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'x', False)
        # Processing the call keyword arguments (line 48)
        kwargs_561081 = {}
        # Getting the type of 'spherical_jn' (line 48)
        spherical_jn_561078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 48)
        spherical_jn_call_result_561082 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), spherical_jn_561078, *[n_561079, x_561080], **kwargs_561081)
        
        
        # Call to array(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_561085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        int_561086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 53), list_561085, int_561086)
        # Adding element type (line 48)
        int_561087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 53), list_561085, int_561087)
        # Adding element type (line 48)
        # Getting the type of 'inf' (line 48)
        inf_561088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 60), 'inf', False)
        int_561089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 65), 'int')
        complex_561090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 67), 'complex')
        # Applying the binary operator '+' (line 48)
        result_add_561091 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 65), '+', int_561089, complex_561090)
        
        # Applying the binary operator '*' (line 48)
        result_mul_561092 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 60), '*', inf_561088, result_add_561091)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 53), list_561085, result_mul_561092)
        
        # Processing the call keyword arguments (line 48)
        kwargs_561093 = {}
        # Getting the type of 'np' (line 48)
        np_561083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 48)
        array_561084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 44), np_561083, 'array')
        # Calling array(args, kwargs) (line 48)
        array_call_result_561094 = invoke(stypy.reporting.localization.Localization(__file__, 48, 44), array_561084, *[list_561085], **kwargs_561093)
        
        # Processing the call keyword arguments (line 48)
        kwargs_561095 = {}
        # Getting the type of 'assert_allclose' (line 48)
        assert_allclose_561077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 48)
        assert_allclose_call_result_561096 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_allclose_561077, *[spherical_jn_call_result_561082, array_call_result_561094], **kwargs_561095)
        
        
        # ################# End of 'test_spherical_jn_inf_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_inf_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_561097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_inf_complex'
        return stypy_return_type_561097


    @norecursion
    def test_spherical_jn_large_arg_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_large_arg_1'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_large_arg_1', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_large_arg_1')
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_large_arg_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_large_arg_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_large_arg_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_large_arg_1(...)' code ##################

        
        # Call to assert_allclose(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Call to spherical_jn(...): (line 54)
        # Processing the call arguments (line 54)
        int_561100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 37), 'int')
        float_561101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'float')
        # Processing the call keyword arguments (line 54)
        kwargs_561102 = {}
        # Getting the type of 'spherical_jn' (line 54)
        spherical_jn_561099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 54)
        spherical_jn_call_result_561103 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), spherical_jn_561099, *[int_561100, float_561101], **kwargs_561102)
        
        float_561104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 51), 'float')
        # Processing the call keyword arguments (line 54)
        kwargs_561105 = {}
        # Getting the type of 'assert_allclose' (line 54)
        assert_allclose_561098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 54)
        assert_allclose_call_result_561106 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_allclose_561098, *[spherical_jn_call_result_561103, float_561104], **kwargs_561105)
        
        
        # ################# End of 'test_spherical_jn_large_arg_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_large_arg_1' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_561107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561107)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_large_arg_1'
        return stypy_return_type_561107


    @norecursion
    def test_spherical_jn_large_arg_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_large_arg_2'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_large_arg_2', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_large_arg_2')
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_large_arg_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_large_arg_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_large_arg_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_large_arg_2(...)' code ##################

        
        # Call to assert_allclose(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to spherical_jn(...): (line 60)
        # Processing the call arguments (line 60)
        int_561110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'int')
        int_561111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 40), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_561112 = {}
        # Getting the type of 'spherical_jn' (line 60)
        spherical_jn_561109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 60)
        spherical_jn_call_result_561113 = invoke(stypy.reporting.localization.Localization(__file__, 60, 24), spherical_jn_561109, *[int_561110, int_561111], **kwargs_561112)
        
        float_561114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 48), 'float')
        # Processing the call keyword arguments (line 60)
        kwargs_561115 = {}
        # Getting the type of 'assert_allclose' (line 60)
        assert_allclose_561108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 60)
        assert_allclose_call_result_561116 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_allclose_561108, *[spherical_jn_call_result_561113, float_561114], **kwargs_561115)
        
        
        # ################# End of 'test_spherical_jn_large_arg_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_large_arg_2' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_561117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_large_arg_2'
        return stypy_return_type_561117


    @norecursion
    def test_spherical_jn_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_at_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_at_zero', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalJn.test_spherical_jn_at_zero')
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJn.test_spherical_jn_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.test_spherical_jn_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_561120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        int_561121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561121)
        # Adding element type (line 65)
        int_561122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561122)
        # Adding element type (line 65)
        int_561123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561123)
        # Adding element type (line 65)
        int_561124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561124)
        # Adding element type (line 65)
        int_561125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561125)
        # Adding element type (line 65)
        int_561126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 21), list_561120, int_561126)
        
        # Processing the call keyword arguments (line 65)
        kwargs_561127 = {}
        # Getting the type of 'np' (line 65)
        np_561118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_561119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), np_561118, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_561128 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), array_561119, *[list_561120], **kwargs_561127)
        
        # Assigning a type to the variable 'n' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'n', array_call_result_561128)
        
        # Assigning a Num to a Name (line 66):
        
        # Assigning a Num to a Name (line 66):
        int_561129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 12), 'int')
        # Assigning a type to the variable 'x' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'x', int_561129)
        
        # Call to assert_allclose(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to spherical_jn(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'n' (line 67)
        n_561132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'n', False)
        # Getting the type of 'x' (line 67)
        x_561133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'x', False)
        # Processing the call keyword arguments (line 67)
        kwargs_561134 = {}
        # Getting the type of 'spherical_jn' (line 67)
        spherical_jn_561131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 67)
        spherical_jn_call_result_561135 = invoke(stypy.reporting.localization.Localization(__file__, 67, 24), spherical_jn_561131, *[n_561132, x_561133], **kwargs_561134)
        
        
        # Call to array(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_561138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        # Adding element type (line 67)
        int_561139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561139)
        # Adding element type (line 67)
        int_561140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561140)
        # Adding element type (line 67)
        int_561141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561141)
        # Adding element type (line 67)
        int_561142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561142)
        # Adding element type (line 67)
        int_561143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561143)
        # Adding element type (line 67)
        int_561144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 53), list_561138, int_561144)
        
        # Processing the call keyword arguments (line 67)
        kwargs_561145 = {}
        # Getting the type of 'np' (line 67)
        np_561136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 67)
        array_561137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 44), np_561136, 'array')
        # Calling array(args, kwargs) (line 67)
        array_call_result_561146 = invoke(stypy.reporting.localization.Localization(__file__, 67, 44), array_561137, *[list_561138], **kwargs_561145)
        
        # Processing the call keyword arguments (line 67)
        kwargs_561147 = {}
        # Getting the type of 'assert_allclose' (line 67)
        assert_allclose_561130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 67)
        assert_allclose_call_result_561148 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), assert_allclose_561130, *[spherical_jn_call_result_561135, array_call_result_561146], **kwargs_561147)
        
        
        # ################# End of 'test_spherical_jn_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_561149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_at_zero'
        return stypy_return_type_561149


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJn.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalJn' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'TestSphericalJn', TestSphericalJn)
# Declaration of the 'TestSphericalYn' class

class TestSphericalYn:

    @norecursion
    def test_spherical_yn_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_exact'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_exact', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_exact')
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_exact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_exact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_exact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_exact(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to array(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_561152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        float_561153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_561152, float_561153)
        # Adding element type (line 75)
        float_561154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_561152, float_561154)
        # Adding element type (line 75)
        float_561155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_561152, float_561155)
        # Adding element type (line 75)
        float_561156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_561152, float_561156)
        # Adding element type (line 75)
        float_561157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 21), list_561152, float_561157)
        
        # Processing the call keyword arguments (line 75)
        kwargs_561158 = {}
        # Getting the type of 'np' (line 75)
        np_561150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 75)
        array_561151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), np_561150, 'array')
        # Calling array(args, kwargs) (line 75)
        array_call_result_561159 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), array_561151, *[list_561152], **kwargs_561158)
        
        # Assigning a type to the variable 'x' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'x', array_call_result_561159)
        
        # Call to assert_allclose(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to spherical_yn(...): (line 76)
        # Processing the call arguments (line 76)
        int_561162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 37), 'int')
        # Getting the type of 'x' (line 76)
        x_561163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 40), 'x', False)
        # Processing the call keyword arguments (line 76)
        kwargs_561164 = {}
        # Getting the type of 'spherical_yn' (line 76)
        spherical_yn_561161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 76)
        spherical_yn_call_result_561165 = invoke(stypy.reporting.localization.Localization(__file__, 76, 24), spherical_yn_561161, *[int_561162, x_561163], **kwargs_561164)
        
        int_561166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
        # Getting the type of 'x' (line 77)
        x_561167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'x', False)
        # Applying the binary operator 'div' (line 77)
        result_div_561168 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 25), 'div', int_561166, x_561167)
        
        int_561169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'int')
        # Getting the type of 'x' (line 77)
        x_561170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 'x', False)
        int_561171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'int')
        # Applying the binary operator '**' (line 77)
        result_pow_561172 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 33), '**', x_561170, int_561171)
        
        # Applying the binary operator 'div' (line 77)
        result_div_561173 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 31), 'div', int_561169, result_pow_561172)
        
        # Applying the binary operator '-' (line 77)
        result_sub_561174 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 25), '-', result_div_561168, result_div_561173)
        
        
        # Call to cos(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'x' (line 77)
        x_561176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 43), 'x', False)
        # Processing the call keyword arguments (line 77)
        kwargs_561177 = {}
        # Getting the type of 'cos' (line 77)
        cos_561175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 39), 'cos', False)
        # Calling cos(args, kwargs) (line 77)
        cos_call_result_561178 = invoke(stypy.reporting.localization.Localization(__file__, 77, 39), cos_561175, *[x_561176], **kwargs_561177)
        
        # Applying the binary operator '*' (line 77)
        result_mul_561179 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 24), '*', result_sub_561174, cos_call_result_561178)
        
        int_561180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 48), 'int')
        # Getting the type of 'x' (line 77)
        x_561181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 50), 'x', False)
        int_561182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 53), 'int')
        # Applying the binary operator '**' (line 77)
        result_pow_561183 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 50), '**', x_561181, int_561182)
        
        # Applying the binary operator 'div' (line 77)
        result_div_561184 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 48), 'div', int_561180, result_pow_561183)
        
        
        # Call to sin(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'x' (line 77)
        x_561186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 59), 'x', False)
        # Processing the call keyword arguments (line 77)
        kwargs_561187 = {}
        # Getting the type of 'sin' (line 77)
        sin_561185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'sin', False)
        # Calling sin(args, kwargs) (line 77)
        sin_call_result_561188 = invoke(stypy.reporting.localization.Localization(__file__, 77, 55), sin_561185, *[x_561186], **kwargs_561187)
        
        # Applying the binary operator '*' (line 77)
        result_mul_561189 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 54), '*', result_div_561184, sin_call_result_561188)
        
        # Applying the binary operator '-' (line 77)
        result_sub_561190 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 24), '-', result_mul_561179, result_mul_561189)
        
        # Processing the call keyword arguments (line 76)
        kwargs_561191 = {}
        # Getting the type of 'assert_allclose' (line 76)
        assert_allclose_561160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 76)
        assert_allclose_call_result_561192 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_allclose_561160, *[spherical_yn_call_result_561165, result_sub_561190], **kwargs_561191)
        
        
        # ################# End of 'test_spherical_yn_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_561193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_exact'
        return stypy_return_type_561193


    @norecursion
    def test_spherical_yn_recurrence_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_recurrence_real'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_recurrence_real', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_recurrence_real')
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_recurrence_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_recurrence_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_recurrence_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_recurrence_real(...)' code ##################

        
        # Assigning a Call to a Name (line 81):
        
        # Assigning a Call to a Name (line 81):
        
        # Call to array(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_561196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        int_561197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_561196, int_561197)
        # Adding element type (line 81)
        int_561198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_561196, int_561198)
        # Adding element type (line 81)
        int_561199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_561196, int_561199)
        # Adding element type (line 81)
        int_561200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_561196, int_561200)
        # Adding element type (line 81)
        int_561201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_561196, int_561201)
        
        # Processing the call keyword arguments (line 81)
        kwargs_561202 = {}
        # Getting the type of 'np' (line 81)
        np_561194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 81)
        array_561195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), np_561194, 'array')
        # Calling array(args, kwargs) (line 81)
        array_call_result_561203 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), array_561195, *[list_561196], **kwargs_561202)
        
        # Assigning a type to the variable 'n' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'n', array_call_result_561203)
        
        # Assigning a Num to a Name (line 82):
        
        # Assigning a Num to a Name (line 82):
        float_561204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'float')
        # Assigning a type to the variable 'x' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'x', float_561204)
        
        # Call to assert_allclose(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to spherical_yn(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'n' (line 83)
        n_561207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 37), 'n', False)
        int_561208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 41), 'int')
        # Applying the binary operator '-' (line 83)
        result_sub_561209 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 37), '-', n_561207, int_561208)
        
        # Getting the type of 'x' (line 83)
        x_561210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'x', False)
        # Processing the call keyword arguments (line 83)
        kwargs_561211 = {}
        # Getting the type of 'spherical_yn' (line 83)
        spherical_yn_561206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 83)
        spherical_yn_call_result_561212 = invoke(stypy.reporting.localization.Localization(__file__, 83, 24), spherical_yn_561206, *[result_sub_561209, x_561210], **kwargs_561211)
        
        
        # Call to spherical_yn(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'n' (line 83)
        n_561214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 62), 'n', False)
        int_561215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 66), 'int')
        # Applying the binary operator '+' (line 83)
        result_add_561216 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 62), '+', n_561214, int_561215)
        
        # Getting the type of 'x' (line 83)
        x_561217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 68), 'x', False)
        # Processing the call keyword arguments (line 83)
        kwargs_561218 = {}
        # Getting the type of 'spherical_yn' (line 83)
        spherical_yn_561213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 83)
        spherical_yn_call_result_561219 = invoke(stypy.reporting.localization.Localization(__file__, 83, 49), spherical_yn_561213, *[result_add_561216, x_561217], **kwargs_561218)
        
        # Applying the binary operator '+' (line 83)
        result_add_561220 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 24), '+', spherical_yn_call_result_561212, spherical_yn_call_result_561219)
        
        int_561221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'int')
        # Getting the type of 'n' (line 84)
        n_561222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'n', False)
        # Applying the binary operator '*' (line 84)
        result_mul_561223 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 25), '*', int_561221, n_561222)
        
        int_561224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'int')
        # Applying the binary operator '+' (line 84)
        result_add_561225 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 25), '+', result_mul_561223, int_561224)
        
        # Getting the type of 'x' (line 84)
        x_561226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 34), 'x', False)
        # Applying the binary operator 'div' (line 84)
        result_div_561227 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 24), 'div', result_add_561225, x_561226)
        
        
        # Call to spherical_yn(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'n' (line 84)
        n_561229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'n', False)
        # Getting the type of 'x' (line 84)
        x_561230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 52), 'x', False)
        # Processing the call keyword arguments (line 84)
        kwargs_561231 = {}
        # Getting the type of 'spherical_yn' (line 84)
        spherical_yn_561228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 84)
        spherical_yn_call_result_561232 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), spherical_yn_561228, *[n_561229, x_561230], **kwargs_561231)
        
        # Applying the binary operator '*' (line 84)
        result_mul_561233 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 35), '*', result_div_561227, spherical_yn_call_result_561232)
        
        # Processing the call keyword arguments (line 83)
        kwargs_561234 = {}
        # Getting the type of 'assert_allclose' (line 83)
        assert_allclose_561205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 83)
        assert_allclose_call_result_561235 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_allclose_561205, *[result_add_561220, result_mul_561233], **kwargs_561234)
        
        
        # ################# End of 'test_spherical_yn_recurrence_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_recurrence_real' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_561236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_recurrence_real'
        return stypy_return_type_561236


    @norecursion
    def test_spherical_yn_recurrence_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_recurrence_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_recurrence_complex', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_recurrence_complex')
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_recurrence_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_recurrence_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_recurrence_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_recurrence_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to array(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_561239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        # Adding element type (line 88)
        int_561240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_561239, int_561240)
        # Adding element type (line 88)
        int_561241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_561239, int_561241)
        # Adding element type (line 88)
        int_561242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_561239, int_561242)
        # Adding element type (line 88)
        int_561243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_561239, int_561243)
        # Adding element type (line 88)
        int_561244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_561239, int_561244)
        
        # Processing the call keyword arguments (line 88)
        kwargs_561245 = {}
        # Getting the type of 'np' (line 88)
        np_561237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 88)
        array_561238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), np_561237, 'array')
        # Calling array(args, kwargs) (line 88)
        array_call_result_561246 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), array_561238, *[list_561239], **kwargs_561245)
        
        # Assigning a type to the variable 'n' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'n', array_call_result_561246)
        
        # Assigning a BinOp to a Name (line 89):
        
        # Assigning a BinOp to a Name (line 89):
        float_561247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'float')
        complex_561248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'complex')
        # Applying the binary operator '+' (line 89)
        result_add_561249 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 12), '+', float_561247, complex_561248)
        
        # Assigning a type to the variable 'x' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'x', result_add_561249)
        
        # Call to assert_allclose(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to spherical_yn(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'n' (line 90)
        n_561252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'n', False)
        int_561253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'int')
        # Applying the binary operator '-' (line 90)
        result_sub_561254 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 37), '-', n_561252, int_561253)
        
        # Getting the type of 'x' (line 90)
        x_561255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 44), 'x', False)
        # Processing the call keyword arguments (line 90)
        kwargs_561256 = {}
        # Getting the type of 'spherical_yn' (line 90)
        spherical_yn_561251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 90)
        spherical_yn_call_result_561257 = invoke(stypy.reporting.localization.Localization(__file__, 90, 24), spherical_yn_561251, *[result_sub_561254, x_561255], **kwargs_561256)
        
        
        # Call to spherical_yn(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'n' (line 90)
        n_561259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 62), 'n', False)
        int_561260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 66), 'int')
        # Applying the binary operator '+' (line 90)
        result_add_561261 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 62), '+', n_561259, int_561260)
        
        # Getting the type of 'x' (line 90)
        x_561262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 69), 'x', False)
        # Processing the call keyword arguments (line 90)
        kwargs_561263 = {}
        # Getting the type of 'spherical_yn' (line 90)
        spherical_yn_561258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 49), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 90)
        spherical_yn_call_result_561264 = invoke(stypy.reporting.localization.Localization(__file__, 90, 49), spherical_yn_561258, *[result_add_561261, x_561262], **kwargs_561263)
        
        # Applying the binary operator '+' (line 90)
        result_add_561265 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 24), '+', spherical_yn_call_result_561257, spherical_yn_call_result_561264)
        
        int_561266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'int')
        # Getting the type of 'n' (line 91)
        n_561267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'n', False)
        # Applying the binary operator '*' (line 91)
        result_mul_561268 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 25), '*', int_561266, n_561267)
        
        int_561269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'int')
        # Applying the binary operator '+' (line 91)
        result_add_561270 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 25), '+', result_mul_561268, int_561269)
        
        # Getting the type of 'x' (line 91)
        x_561271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'x', False)
        # Applying the binary operator 'div' (line 91)
        result_div_561272 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 24), 'div', result_add_561270, x_561271)
        
        
        # Call to spherical_yn(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'n' (line 91)
        n_561274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'n', False)
        # Getting the type of 'x' (line 91)
        x_561275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 52), 'x', False)
        # Processing the call keyword arguments (line 91)
        kwargs_561276 = {}
        # Getting the type of 'spherical_yn' (line 91)
        spherical_yn_561273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 91)
        spherical_yn_call_result_561277 = invoke(stypy.reporting.localization.Localization(__file__, 91, 36), spherical_yn_561273, *[n_561274, x_561275], **kwargs_561276)
        
        # Applying the binary operator '*' (line 91)
        result_mul_561278 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 35), '*', result_div_561272, spherical_yn_call_result_561277)
        
        # Processing the call keyword arguments (line 90)
        kwargs_561279 = {}
        # Getting the type of 'assert_allclose' (line 90)
        assert_allclose_561250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 90)
        assert_allclose_call_result_561280 = invoke(stypy.reporting.localization.Localization(__file__, 90, 8), assert_allclose_561250, *[result_add_561265, result_mul_561278], **kwargs_561279)
        
        
        # ################# End of 'test_spherical_yn_recurrence_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_recurrence_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_561281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_recurrence_complex'
        return stypy_return_type_561281


    @norecursion
    def test_spherical_yn_inf_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_inf_real'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_inf_real', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_inf_real')
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_inf_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_inf_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_inf_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_inf_real(...)' code ##################

        
        # Assigning a Num to a Name (line 95):
        
        # Assigning a Num to a Name (line 95):
        int_561282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 12), 'int')
        # Assigning a type to the variable 'n' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'n', int_561282)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to array(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_561285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Getting the type of 'inf' (line 96)
        inf_561286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 96)
        result___neg___561287 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), 'usub', inf_561286)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 21), list_561285, result___neg___561287)
        # Adding element type (line 96)
        # Getting the type of 'inf' (line 96)
        inf_561288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 21), list_561285, inf_561288)
        
        # Processing the call keyword arguments (line 96)
        kwargs_561289 = {}
        # Getting the type of 'np' (line 96)
        np_561283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 96)
        array_561284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), np_561283, 'array')
        # Calling array(args, kwargs) (line 96)
        array_call_result_561290 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), array_561284, *[list_561285], **kwargs_561289)
        
        # Assigning a type to the variable 'x' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'x', array_call_result_561290)
        
        # Call to assert_allclose(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to spherical_yn(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'n' (line 97)
        n_561293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'n', False)
        # Getting the type of 'x' (line 97)
        x_561294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'x', False)
        # Processing the call keyword arguments (line 97)
        kwargs_561295 = {}
        # Getting the type of 'spherical_yn' (line 97)
        spherical_yn_561292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 97)
        spherical_yn_call_result_561296 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), spherical_yn_561292, *[n_561293, x_561294], **kwargs_561295)
        
        
        # Call to array(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_561299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        int_561300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 53), list_561299, int_561300)
        # Adding element type (line 97)
        int_561301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 53), list_561299, int_561301)
        
        # Processing the call keyword arguments (line 97)
        kwargs_561302 = {}
        # Getting the type of 'np' (line 97)
        np_561297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 97)
        array_561298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 44), np_561297, 'array')
        # Calling array(args, kwargs) (line 97)
        array_call_result_561303 = invoke(stypy.reporting.localization.Localization(__file__, 97, 44), array_561298, *[list_561299], **kwargs_561302)
        
        # Processing the call keyword arguments (line 97)
        kwargs_561304 = {}
        # Getting the type of 'assert_allclose' (line 97)
        assert_allclose_561291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 97)
        assert_allclose_call_result_561305 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), assert_allclose_561291, *[spherical_yn_call_result_561296, array_call_result_561303], **kwargs_561304)
        
        
        # ################# End of 'test_spherical_yn_inf_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_inf_real' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_561306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_inf_real'
        return stypy_return_type_561306


    @norecursion
    def test_spherical_yn_inf_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_inf_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_inf_complex', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_inf_complex')
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_inf_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_inf_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_inf_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_inf_complex(...)' code ##################

        
        # Assigning a Num to a Name (line 101):
        
        # Assigning a Num to a Name (line 101):
        int_561307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'int')
        # Assigning a type to the variable 'n' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'n', int_561307)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to array(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_561310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        # Adding element type (line 102)
        
        # Getting the type of 'inf' (line 102)
        inf_561311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 102)
        result___neg___561312 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 22), 'usub', inf_561311)
        
        complex_561313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'complex')
        # Applying the binary operator '+' (line 102)
        result_add_561314 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 22), '+', result___neg___561312, complex_561313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), list_561310, result_add_561314)
        # Adding element type (line 102)
        # Getting the type of 'inf' (line 102)
        inf_561315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 33), 'inf', False)
        complex_561316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 39), 'complex')
        # Applying the binary operator '+' (line 102)
        result_add_561317 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 33), '+', inf_561315, complex_561316)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), list_561310, result_add_561317)
        # Adding element type (line 102)
        # Getting the type of 'inf' (line 102)
        inf_561318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'inf', False)
        int_561319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 48), 'int')
        complex_561320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'complex')
        # Applying the binary operator '+' (line 102)
        result_add_561321 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 48), '+', int_561319, complex_561320)
        
        # Applying the binary operator '*' (line 102)
        result_mul_561322 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 43), '*', inf_561318, result_add_561321)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), list_561310, result_mul_561322)
        
        # Processing the call keyword arguments (line 102)
        kwargs_561323 = {}
        # Getting the type of 'np' (line 102)
        np_561308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 102)
        array_561309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), np_561308, 'array')
        # Calling array(args, kwargs) (line 102)
        array_call_result_561324 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), array_561309, *[list_561310], **kwargs_561323)
        
        # Assigning a type to the variable 'x' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'x', array_call_result_561324)
        
        # Call to assert_allclose(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to spherical_yn(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'n' (line 103)
        n_561327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'n', False)
        # Getting the type of 'x' (line 103)
        x_561328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'x', False)
        # Processing the call keyword arguments (line 103)
        kwargs_561329 = {}
        # Getting the type of 'spherical_yn' (line 103)
        spherical_yn_561326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 103)
        spherical_yn_call_result_561330 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), spherical_yn_561326, *[n_561327, x_561328], **kwargs_561329)
        
        
        # Call to array(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'list' (line 103)
        list_561333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 103)
        # Adding element type (line 103)
        int_561334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 53), list_561333, int_561334)
        # Adding element type (line 103)
        int_561335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 53), list_561333, int_561335)
        # Adding element type (line 103)
        # Getting the type of 'inf' (line 103)
        inf_561336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 60), 'inf', False)
        int_561337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 65), 'int')
        complex_561338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 67), 'complex')
        # Applying the binary operator '+' (line 103)
        result_add_561339 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 65), '+', int_561337, complex_561338)
        
        # Applying the binary operator '*' (line 103)
        result_mul_561340 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 60), '*', inf_561336, result_add_561339)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 53), list_561333, result_mul_561340)
        
        # Processing the call keyword arguments (line 103)
        kwargs_561341 = {}
        # Getting the type of 'np' (line 103)
        np_561331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 103)
        array_561332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 44), np_561331, 'array')
        # Calling array(args, kwargs) (line 103)
        array_call_result_561342 = invoke(stypy.reporting.localization.Localization(__file__, 103, 44), array_561332, *[list_561333], **kwargs_561341)
        
        # Processing the call keyword arguments (line 103)
        kwargs_561343 = {}
        # Getting the type of 'assert_allclose' (line 103)
        assert_allclose_561325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 103)
        assert_allclose_call_result_561344 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), assert_allclose_561325, *[spherical_yn_call_result_561330, array_call_result_561342], **kwargs_561343)
        
        
        # ################# End of 'test_spherical_yn_inf_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_inf_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_561345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561345)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_inf_complex'
        return stypy_return_type_561345


    @norecursion
    def test_spherical_yn_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_at_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_at_zero', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_at_zero')
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to array(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_561348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        int_561349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561349)
        # Adding element type (line 107)
        int_561350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561350)
        # Adding element type (line 107)
        int_561351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561351)
        # Adding element type (line 107)
        int_561352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561352)
        # Adding element type (line 107)
        int_561353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561353)
        # Adding element type (line 107)
        int_561354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), list_561348, int_561354)
        
        # Processing the call keyword arguments (line 107)
        kwargs_561355 = {}
        # Getting the type of 'np' (line 107)
        np_561346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 107)
        array_561347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), np_561346, 'array')
        # Calling array(args, kwargs) (line 107)
        array_call_result_561356 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), array_561347, *[list_561348], **kwargs_561355)
        
        # Assigning a type to the variable 'n' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'n', array_call_result_561356)
        
        # Assigning a Num to a Name (line 108):
        
        # Assigning a Num to a Name (line 108):
        int_561357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'int')
        # Assigning a type to the variable 'x' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'x', int_561357)
        
        # Call to assert_allclose(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to spherical_yn(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'n' (line 109)
        n_561360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'n', False)
        # Getting the type of 'x' (line 109)
        x_561361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'x', False)
        # Processing the call keyword arguments (line 109)
        kwargs_561362 = {}
        # Getting the type of 'spherical_yn' (line 109)
        spherical_yn_561359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 109)
        spherical_yn_call_result_561363 = invoke(stypy.reporting.localization.Localization(__file__, 109, 24), spherical_yn_561359, *[n_561360, x_561361], **kwargs_561362)
        
        
        # Getting the type of 'inf' (line 109)
        inf_561364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 45), 'inf', False)
        # Applying the 'usub' unary operator (line 109)
        result___neg___561365 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 44), 'usub', inf_561364)
        
        
        # Call to ones(...): (line 109)
        # Processing the call keyword arguments (line 109)
        # Getting the type of 'n' (line 109)
        n_561368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 63), 'n', False)
        # Obtaining the member 'shape' of a type (line 109)
        shape_561369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 63), n_561368, 'shape')
        keyword_561370 = shape_561369
        kwargs_561371 = {'shape': keyword_561370}
        # Getting the type of 'np' (line 109)
        np_561366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'np', False)
        # Obtaining the member 'ones' of a type (line 109)
        ones_561367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 49), np_561366, 'ones')
        # Calling ones(args, kwargs) (line 109)
        ones_call_result_561372 = invoke(stypy.reporting.localization.Localization(__file__, 109, 49), ones_561367, *[], **kwargs_561371)
        
        # Applying the binary operator '*' (line 109)
        result_mul_561373 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 44), '*', result___neg___561365, ones_call_result_561372)
        
        # Processing the call keyword arguments (line 109)
        kwargs_561374 = {}
        # Getting the type of 'assert_allclose' (line 109)
        assert_allclose_561358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 109)
        assert_allclose_call_result_561375 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert_allclose_561358, *[spherical_yn_call_result_561363, result_mul_561373], **kwargs_561374)
        
        
        # ################# End of 'test_spherical_yn_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_561376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561376)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_at_zero'
        return stypy_return_type_561376


    @norecursion
    def test_spherical_yn_at_zero_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_yn_at_zero_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_yn_at_zero_complex', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalYn.test_spherical_yn_at_zero_complex')
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYn.test_spherical_yn_at_zero_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.test_spherical_yn_at_zero_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_yn_at_zero_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_yn_at_zero_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to array(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_561379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        int_561380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561380)
        # Adding element type (line 117)
        int_561381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561381)
        # Adding element type (line 117)
        int_561382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561382)
        # Adding element type (line 117)
        int_561383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561383)
        # Adding element type (line 117)
        int_561384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561384)
        # Adding element type (line 117)
        int_561385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 21), list_561379, int_561385)
        
        # Processing the call keyword arguments (line 117)
        kwargs_561386 = {}
        # Getting the type of 'np' (line 117)
        np_561377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 117)
        array_561378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), np_561377, 'array')
        # Calling array(args, kwargs) (line 117)
        array_call_result_561387 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), array_561378, *[list_561379], **kwargs_561386)
        
        # Assigning a type to the variable 'n' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'n', array_call_result_561387)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        int_561388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'int')
        complex_561389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'complex')
        # Applying the binary operator '+' (line 118)
        result_add_561390 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 12), '+', int_561388, complex_561389)
        
        # Assigning a type to the variable 'x' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'x', result_add_561390)
        
        # Call to assert_allclose(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to spherical_yn(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'n' (line 119)
        n_561393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'n', False)
        # Getting the type of 'x' (line 119)
        x_561394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'x', False)
        # Processing the call keyword arguments (line 119)
        kwargs_561395 = {}
        # Getting the type of 'spherical_yn' (line 119)
        spherical_yn_561392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 119)
        spherical_yn_call_result_561396 = invoke(stypy.reporting.localization.Localization(__file__, 119, 24), spherical_yn_561392, *[n_561393, x_561394], **kwargs_561395)
        
        # Getting the type of 'nan' (line 119)
        nan_561397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 44), 'nan', False)
        
        # Call to ones(...): (line 119)
        # Processing the call keyword arguments (line 119)
        # Getting the type of 'n' (line 119)
        n_561400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 62), 'n', False)
        # Obtaining the member 'shape' of a type (line 119)
        shape_561401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 62), n_561400, 'shape')
        keyword_561402 = shape_561401
        kwargs_561403 = {'shape': keyword_561402}
        # Getting the type of 'np' (line 119)
        np_561398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'np', False)
        # Obtaining the member 'ones' of a type (line 119)
        ones_561399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 48), np_561398, 'ones')
        # Calling ones(args, kwargs) (line 119)
        ones_call_result_561404 = invoke(stypy.reporting.localization.Localization(__file__, 119, 48), ones_561399, *[], **kwargs_561403)
        
        # Applying the binary operator '*' (line 119)
        result_mul_561405 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 44), '*', nan_561397, ones_call_result_561404)
        
        # Processing the call keyword arguments (line 119)
        kwargs_561406 = {}
        # Getting the type of 'assert_allclose' (line 119)
        assert_allclose_561391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 119)
        assert_allclose_call_result_561407 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assert_allclose_561391, *[spherical_yn_call_result_561396, result_mul_561405], **kwargs_561406)
        
        
        # ################# End of 'test_spherical_yn_at_zero_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_yn_at_zero_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_561408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_yn_at_zero_complex'
        return stypy_return_type_561408


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 70, 0, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYn.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalYn' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'TestSphericalYn', TestSphericalYn)
# Declaration of the 'TestSphericalJnYnCrossProduct' class

class TestSphericalJnYnCrossProduct:

    @norecursion
    def test_spherical_jn_yn_cross_product_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_yn_cross_product_1'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_yn_cross_product_1', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_function_name', 'TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1')
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_yn_cross_product_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_yn_cross_product_1(...)' code ##################

        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to array(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_561411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        int_561412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 21), list_561411, int_561412)
        # Adding element type (line 125)
        int_561413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 21), list_561411, int_561413)
        # Adding element type (line 125)
        int_561414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 21), list_561411, int_561414)
        
        # Processing the call keyword arguments (line 125)
        kwargs_561415 = {}
        # Getting the type of 'np' (line 125)
        np_561409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 125)
        array_561410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), np_561409, 'array')
        # Calling array(args, kwargs) (line 125)
        array_call_result_561416 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), array_561410, *[list_561411], **kwargs_561415)
        
        # Assigning a type to the variable 'n' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'n', array_call_result_561416)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to array(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_561419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        float_561420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 21), list_561419, float_561420)
        # Adding element type (line 126)
        int_561421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 21), list_561419, int_561421)
        # Adding element type (line 126)
        int_561422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 21), list_561419, int_561422)
        
        # Processing the call keyword arguments (line 126)
        kwargs_561423 = {}
        # Getting the type of 'np' (line 126)
        np_561417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 126)
        array_561418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), np_561417, 'array')
        # Calling array(args, kwargs) (line 126)
        array_call_result_561424 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), array_561418, *[list_561419], **kwargs_561423)
        
        # Assigning a type to the variable 'x' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x', array_call_result_561424)
        
        # Assigning a BinOp to a Name (line 127):
        
        # Assigning a BinOp to a Name (line 127):
        
        # Call to spherical_jn(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'n' (line 127)
        n_561426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'n', False)
        int_561427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 33), 'int')
        # Applying the binary operator '+' (line 127)
        result_add_561428 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 29), '+', n_561426, int_561427)
        
        # Getting the type of 'x' (line 127)
        x_561429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 36), 'x', False)
        # Processing the call keyword arguments (line 127)
        kwargs_561430 = {}
        # Getting the type of 'spherical_jn' (line 127)
        spherical_jn_561425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 127)
        spherical_jn_call_result_561431 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), spherical_jn_561425, *[result_add_561428, x_561429], **kwargs_561430)
        
        
        # Call to spherical_yn(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'n' (line 127)
        n_561433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'n', False)
        # Getting the type of 'x' (line 127)
        x_561434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 57), 'x', False)
        # Processing the call keyword arguments (line 127)
        kwargs_561435 = {}
        # Getting the type of 'spherical_yn' (line 127)
        spherical_yn_561432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 127)
        spherical_yn_call_result_561436 = invoke(stypy.reporting.localization.Localization(__file__, 127, 41), spherical_yn_561432, *[n_561433, x_561434], **kwargs_561435)
        
        # Applying the binary operator '*' (line 127)
        result_mul_561437 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '*', spherical_jn_call_result_561431, spherical_yn_call_result_561436)
        
        
        # Call to spherical_jn(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'n' (line 128)
        n_561439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'n', False)
        # Getting the type of 'x' (line 128)
        x_561440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'x', False)
        # Processing the call keyword arguments (line 128)
        kwargs_561441 = {}
        # Getting the type of 'spherical_jn' (line 128)
        spherical_jn_561438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 128)
        spherical_jn_call_result_561442 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), spherical_jn_561438, *[n_561439, x_561440], **kwargs_561441)
        
        
        # Call to spherical_yn(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'n' (line 128)
        n_561444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'n', False)
        int_561445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 54), 'int')
        # Applying the binary operator '+' (line 128)
        result_add_561446 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 50), '+', n_561444, int_561445)
        
        # Getting the type of 'x' (line 128)
        x_561447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 57), 'x', False)
        # Processing the call keyword arguments (line 128)
        kwargs_561448 = {}
        # Getting the type of 'spherical_yn' (line 128)
        spherical_yn_561443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 37), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 128)
        spherical_yn_call_result_561449 = invoke(stypy.reporting.localization.Localization(__file__, 128, 37), spherical_yn_561443, *[result_add_561446, x_561447], **kwargs_561448)
        
        # Applying the binary operator '*' (line 128)
        result_mul_561450 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 16), '*', spherical_jn_call_result_561442, spherical_yn_call_result_561449)
        
        # Applying the binary operator '-' (line 127)
        result_sub_561451 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '-', result_mul_561437, result_mul_561450)
        
        # Assigning a type to the variable 'left' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'left', result_sub_561451)
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        int_561452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 16), 'int')
        # Getting the type of 'x' (line 129)
        x_561453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 18), 'x')
        int_561454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 21), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_561455 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 18), '**', x_561453, int_561454)
        
        # Applying the binary operator 'div' (line 129)
        result_div_561456 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), 'div', int_561452, result_pow_561455)
        
        # Assigning a type to the variable 'right' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'right', result_div_561456)
        
        # Call to assert_allclose(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'left' (line 130)
        left_561458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'left', False)
        # Getting the type of 'right' (line 130)
        right_561459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'right', False)
        # Processing the call keyword arguments (line 130)
        kwargs_561460 = {}
        # Getting the type of 'assert_allclose' (line 130)
        assert_allclose_561457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 130)
        assert_allclose_call_result_561461 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assert_allclose_561457, *[left_561458, right_561459], **kwargs_561460)
        
        
        # ################# End of 'test_spherical_jn_yn_cross_product_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_yn_cross_product_1' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_561462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_yn_cross_product_1'
        return stypy_return_type_561462


    @norecursion
    def test_spherical_jn_yn_cross_product_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_yn_cross_product_2'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_yn_cross_product_2', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_function_name', 'TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2')
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnYnCrossProduct.test_spherical_jn_yn_cross_product_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_yn_cross_product_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_yn_cross_product_2(...)' code ##################

        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to array(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_561465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        int_561466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), list_561465, int_561466)
        # Adding element type (line 134)
        int_561467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), list_561465, int_561467)
        # Adding element type (line 134)
        int_561468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 21), list_561465, int_561468)
        
        # Processing the call keyword arguments (line 134)
        kwargs_561469 = {}
        # Getting the type of 'np' (line 134)
        np_561463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 134)
        array_561464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), np_561463, 'array')
        # Calling array(args, kwargs) (line 134)
        array_call_result_561470 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), array_561464, *[list_561465], **kwargs_561469)
        
        # Assigning a type to the variable 'n' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'n', array_call_result_561470)
        
        # Assigning a Call to a Name (line 135):
        
        # Assigning a Call to a Name (line 135):
        
        # Call to array(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_561473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        float_561474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_561473, float_561474)
        # Adding element type (line 135)
        int_561475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_561473, int_561475)
        # Adding element type (line 135)
        int_561476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 21), list_561473, int_561476)
        
        # Processing the call keyword arguments (line 135)
        kwargs_561477 = {}
        # Getting the type of 'np' (line 135)
        np_561471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 135)
        array_561472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), np_561471, 'array')
        # Calling array(args, kwargs) (line 135)
        array_call_result_561478 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), array_561472, *[list_561473], **kwargs_561477)
        
        # Assigning a type to the variable 'x' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'x', array_call_result_561478)
        
        # Assigning a BinOp to a Name (line 136):
        
        # Assigning a BinOp to a Name (line 136):
        
        # Call to spherical_jn(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'n' (line 136)
        n_561480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'n', False)
        int_561481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 33), 'int')
        # Applying the binary operator '+' (line 136)
        result_add_561482 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 29), '+', n_561480, int_561481)
        
        # Getting the type of 'x' (line 136)
        x_561483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), 'x', False)
        # Processing the call keyword arguments (line 136)
        kwargs_561484 = {}
        # Getting the type of 'spherical_jn' (line 136)
        spherical_jn_561479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 136)
        spherical_jn_call_result_561485 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), spherical_jn_561479, *[result_add_561482, x_561483], **kwargs_561484)
        
        
        # Call to spherical_yn(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'n' (line 136)
        n_561487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 54), 'n', False)
        # Getting the type of 'x' (line 136)
        x_561488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 57), 'x', False)
        # Processing the call keyword arguments (line 136)
        kwargs_561489 = {}
        # Getting the type of 'spherical_yn' (line 136)
        spherical_yn_561486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 41), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 136)
        spherical_yn_call_result_561490 = invoke(stypy.reporting.localization.Localization(__file__, 136, 41), spherical_yn_561486, *[n_561487, x_561488], **kwargs_561489)
        
        # Applying the binary operator '*' (line 136)
        result_mul_561491 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '*', spherical_jn_call_result_561485, spherical_yn_call_result_561490)
        
        
        # Call to spherical_jn(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'n' (line 137)
        n_561493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'n', False)
        # Getting the type of 'x' (line 137)
        x_561494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'x', False)
        # Processing the call keyword arguments (line 137)
        kwargs_561495 = {}
        # Getting the type of 'spherical_jn' (line 137)
        spherical_jn_561492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 137)
        spherical_jn_call_result_561496 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), spherical_jn_561492, *[n_561493, x_561494], **kwargs_561495)
        
        
        # Call to spherical_yn(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'n' (line 137)
        n_561498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'n', False)
        int_561499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 54), 'int')
        # Applying the binary operator '+' (line 137)
        result_add_561500 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 50), '+', n_561498, int_561499)
        
        # Getting the type of 'x' (line 137)
        x_561501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 57), 'x', False)
        # Processing the call keyword arguments (line 137)
        kwargs_561502 = {}
        # Getting the type of 'spherical_yn' (line 137)
        spherical_yn_561497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 137)
        spherical_yn_call_result_561503 = invoke(stypy.reporting.localization.Localization(__file__, 137, 37), spherical_yn_561497, *[result_add_561500, x_561501], **kwargs_561502)
        
        # Applying the binary operator '*' (line 137)
        result_mul_561504 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), '*', spherical_jn_call_result_561496, spherical_yn_call_result_561503)
        
        # Applying the binary operator '-' (line 136)
        result_sub_561505 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 16), '-', result_mul_561491, result_mul_561504)
        
        # Assigning a type to the variable 'left' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'left', result_sub_561505)
        
        # Assigning a BinOp to a Name (line 138):
        
        # Assigning a BinOp to a Name (line 138):
        int_561506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 17), 'int')
        # Getting the type of 'n' (line 138)
        n_561507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'n')
        # Applying the binary operator '*' (line 138)
        result_mul_561508 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 17), '*', int_561506, n_561507)
        
        int_561509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'int')
        # Applying the binary operator '+' (line 138)
        result_add_561510 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 17), '+', result_mul_561508, int_561509)
        
        # Getting the type of 'x' (line 138)
        x_561511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 26), 'x')
        int_561512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'int')
        # Applying the binary operator '**' (line 138)
        result_pow_561513 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 26), '**', x_561511, int_561512)
        
        # Applying the binary operator 'div' (line 138)
        result_div_561514 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 16), 'div', result_add_561510, result_pow_561513)
        
        # Assigning a type to the variable 'right' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'right', result_div_561514)
        
        # Call to assert_allclose(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'left' (line 139)
        left_561516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'left', False)
        # Getting the type of 'right' (line 139)
        right_561517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'right', False)
        # Processing the call keyword arguments (line 139)
        kwargs_561518 = {}
        # Getting the type of 'assert_allclose' (line 139)
        assert_allclose_561515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 139)
        assert_allclose_call_result_561519 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assert_allclose_561515, *[left_561516, right_561517], **kwargs_561518)
        
        
        # ################# End of 'test_spherical_jn_yn_cross_product_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_yn_cross_product_2' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_561520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561520)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_yn_cross_product_2'
        return stypy_return_type_561520


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 122, 0, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnYnCrossProduct.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalJnYnCrossProduct' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'TestSphericalJnYnCrossProduct', TestSphericalJnYnCrossProduct)
# Declaration of the 'TestSphericalIn' class

class TestSphericalIn:

    @norecursion
    def test_spherical_in_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_exact'
        module_type_store = module_type_store.open_function_context('test_spherical_in_exact', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_exact')
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_exact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_exact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_exact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_exact(...)' code ##################

        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to array(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_561523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        # Adding element type (line 145)
        float_561524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), list_561523, float_561524)
        # Adding element type (line 145)
        float_561525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), list_561523, float_561525)
        # Adding element type (line 145)
        float_561526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), list_561523, float_561526)
        # Adding element type (line 145)
        float_561527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), list_561523, float_561527)
        
        # Processing the call keyword arguments (line 145)
        kwargs_561528 = {}
        # Getting the type of 'np' (line 145)
        np_561521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 145)
        array_561522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), np_561521, 'array')
        # Calling array(args, kwargs) (line 145)
        array_call_result_561529 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), array_561522, *[list_561523], **kwargs_561528)
        
        # Assigning a type to the variable 'x' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'x', array_call_result_561529)
        
        # Call to assert_allclose(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to spherical_in(...): (line 146)
        # Processing the call arguments (line 146)
        int_561532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 37), 'int')
        # Getting the type of 'x' (line 146)
        x_561533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'x', False)
        # Processing the call keyword arguments (line 146)
        kwargs_561534 = {}
        # Getting the type of 'spherical_in' (line 146)
        spherical_in_561531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 146)
        spherical_in_call_result_561535 = invoke(stypy.reporting.localization.Localization(__file__, 146, 24), spherical_in_561531, *[int_561532, x_561533], **kwargs_561534)
        
        int_561536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'int')
        # Getting the type of 'x' (line 147)
        x_561537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'x', False)
        # Applying the binary operator 'div' (line 147)
        result_div_561538 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 25), 'div', int_561536, x_561537)
        
        int_561539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'int')
        # Getting the type of 'x' (line 147)
        x_561540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'x', False)
        int_561541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 36), 'int')
        # Applying the binary operator '**' (line 147)
        result_pow_561542 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 33), '**', x_561540, int_561541)
        
        # Applying the binary operator 'div' (line 147)
        result_div_561543 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 31), 'div', int_561539, result_pow_561542)
        
        # Applying the binary operator '+' (line 147)
        result_add_561544 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 25), '+', result_div_561538, result_div_561543)
        
        
        # Call to sinh(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'x' (line 147)
        x_561546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 44), 'x', False)
        # Processing the call keyword arguments (line 147)
        kwargs_561547 = {}
        # Getting the type of 'sinh' (line 147)
        sinh_561545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 39), 'sinh', False)
        # Calling sinh(args, kwargs) (line 147)
        sinh_call_result_561548 = invoke(stypy.reporting.localization.Localization(__file__, 147, 39), sinh_561545, *[x_561546], **kwargs_561547)
        
        # Applying the binary operator '*' (line 147)
        result_mul_561549 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 24), '*', result_add_561544, sinh_call_result_561548)
        
        int_561550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 49), 'int')
        # Getting the type of 'x' (line 147)
        x_561551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'x', False)
        int_561552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 54), 'int')
        # Applying the binary operator '**' (line 147)
        result_pow_561553 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 51), '**', x_561551, int_561552)
        
        # Applying the binary operator 'div' (line 147)
        result_div_561554 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 49), 'div', int_561550, result_pow_561553)
        
        
        # Call to cosh(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'x' (line 147)
        x_561556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'x', False)
        # Processing the call keyword arguments (line 147)
        kwargs_561557 = {}
        # Getting the type of 'cosh' (line 147)
        cosh_561555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 56), 'cosh', False)
        # Calling cosh(args, kwargs) (line 147)
        cosh_call_result_561558 = invoke(stypy.reporting.localization.Localization(__file__, 147, 56), cosh_561555, *[x_561556], **kwargs_561557)
        
        # Applying the binary operator '*' (line 147)
        result_mul_561559 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 55), '*', result_div_561554, cosh_call_result_561558)
        
        # Applying the binary operator '-' (line 147)
        result_sub_561560 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 24), '-', result_mul_561549, result_mul_561559)
        
        # Processing the call keyword arguments (line 146)
        kwargs_561561 = {}
        # Getting the type of 'assert_allclose' (line 146)
        assert_allclose_561530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 146)
        assert_allclose_call_result_561562 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assert_allclose_561530, *[spherical_in_call_result_561535, result_sub_561560], **kwargs_561561)
        
        
        # ################# End of 'test_spherical_in_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_561563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561563)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_exact'
        return stypy_return_type_561563


    @norecursion
    def test_spherical_in_recurrence_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_recurrence_real'
        module_type_store = module_type_store.open_function_context('test_spherical_in_recurrence_real', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_recurrence_real')
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_recurrence_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_recurrence_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_recurrence_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_recurrence_real(...)' code ##################

        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to array(...): (line 151)
        # Processing the call arguments (line 151)
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_561566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        int_561567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 21), list_561566, int_561567)
        # Adding element type (line 151)
        int_561568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 21), list_561566, int_561568)
        # Adding element type (line 151)
        int_561569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 21), list_561566, int_561569)
        # Adding element type (line 151)
        int_561570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 21), list_561566, int_561570)
        # Adding element type (line 151)
        int_561571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 21), list_561566, int_561571)
        
        # Processing the call keyword arguments (line 151)
        kwargs_561572 = {}
        # Getting the type of 'np' (line 151)
        np_561564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 151)
        array_561565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), np_561564, 'array')
        # Calling array(args, kwargs) (line 151)
        array_call_result_561573 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), array_561565, *[list_561566], **kwargs_561572)
        
        # Assigning a type to the variable 'n' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'n', array_call_result_561573)
        
        # Assigning a Num to a Name (line 152):
        
        # Assigning a Num to a Name (line 152):
        float_561574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 12), 'float')
        # Assigning a type to the variable 'x' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'x', float_561574)
        
        # Call to assert_allclose(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to spherical_in(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'n' (line 153)
        n_561577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'n', False)
        int_561578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 41), 'int')
        # Applying the binary operator '-' (line 153)
        result_sub_561579 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 37), '-', n_561577, int_561578)
        
        # Getting the type of 'x' (line 153)
        x_561580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 44), 'x', False)
        # Processing the call keyword arguments (line 153)
        kwargs_561581 = {}
        # Getting the type of 'spherical_in' (line 153)
        spherical_in_561576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 153)
        spherical_in_call_result_561582 = invoke(stypy.reporting.localization.Localization(__file__, 153, 24), spherical_in_561576, *[result_sub_561579, x_561580], **kwargs_561581)
        
        
        # Call to spherical_in(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'n' (line 153)
        n_561584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 62), 'n', False)
        int_561585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 66), 'int')
        # Applying the binary operator '+' (line 153)
        result_add_561586 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 62), '+', n_561584, int_561585)
        
        # Getting the type of 'x' (line 153)
        x_561587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 68), 'x', False)
        # Processing the call keyword arguments (line 153)
        kwargs_561588 = {}
        # Getting the type of 'spherical_in' (line 153)
        spherical_in_561583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 49), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 153)
        spherical_in_call_result_561589 = invoke(stypy.reporting.localization.Localization(__file__, 153, 49), spherical_in_561583, *[result_add_561586, x_561587], **kwargs_561588)
        
        # Applying the binary operator '-' (line 153)
        result_sub_561590 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 24), '-', spherical_in_call_result_561582, spherical_in_call_result_561589)
        
        int_561591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'int')
        # Getting the type of 'n' (line 154)
        n_561592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'n', False)
        # Applying the binary operator '*' (line 154)
        result_mul_561593 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 25), '*', int_561591, n_561592)
        
        int_561594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 31), 'int')
        # Applying the binary operator '+' (line 154)
        result_add_561595 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 25), '+', result_mul_561593, int_561594)
        
        # Getting the type of 'x' (line 154)
        x_561596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'x', False)
        # Applying the binary operator 'div' (line 154)
        result_div_561597 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 24), 'div', result_add_561595, x_561596)
        
        
        # Call to spherical_in(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'n' (line 154)
        n_561599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'n', False)
        # Getting the type of 'x' (line 154)
        x_561600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 52), 'x', False)
        # Processing the call keyword arguments (line 154)
        kwargs_561601 = {}
        # Getting the type of 'spherical_in' (line 154)
        spherical_in_561598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 154)
        spherical_in_call_result_561602 = invoke(stypy.reporting.localization.Localization(__file__, 154, 36), spherical_in_561598, *[n_561599, x_561600], **kwargs_561601)
        
        # Applying the binary operator '*' (line 154)
        result_mul_561603 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 35), '*', result_div_561597, spherical_in_call_result_561602)
        
        # Processing the call keyword arguments (line 153)
        kwargs_561604 = {}
        # Getting the type of 'assert_allclose' (line 153)
        assert_allclose_561575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 153)
        assert_allclose_call_result_561605 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_allclose_561575, *[result_sub_561590, result_mul_561603], **kwargs_561604)
        
        
        # ################# End of 'test_spherical_in_recurrence_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_recurrence_real' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_561606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561606)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_recurrence_real'
        return stypy_return_type_561606


    @norecursion
    def test_spherical_in_recurrence_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_recurrence_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_in_recurrence_complex', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_recurrence_complex')
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_recurrence_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_recurrence_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_recurrence_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_recurrence_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 158):
        
        # Assigning a Call to a Name (line 158):
        
        # Call to array(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_561609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        int_561610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_561609, int_561610)
        # Adding element type (line 158)
        int_561611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_561609, int_561611)
        # Adding element type (line 158)
        int_561612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_561609, int_561612)
        # Adding element type (line 158)
        int_561613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_561609, int_561613)
        # Adding element type (line 158)
        int_561614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 21), list_561609, int_561614)
        
        # Processing the call keyword arguments (line 158)
        kwargs_561615 = {}
        # Getting the type of 'np' (line 158)
        np_561607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 158)
        array_561608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), np_561607, 'array')
        # Calling array(args, kwargs) (line 158)
        array_call_result_561616 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), array_561608, *[list_561609], **kwargs_561615)
        
        # Assigning a type to the variable 'n' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'n', array_call_result_561616)
        
        # Assigning a BinOp to a Name (line 159):
        
        # Assigning a BinOp to a Name (line 159):
        float_561617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 12), 'float')
        complex_561618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'complex')
        # Applying the binary operator '+' (line 159)
        result_add_561619 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 12), '+', float_561617, complex_561618)
        
        # Assigning a type to the variable 'x' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'x', result_add_561619)
        
        # Call to assert_allclose(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to spherical_in(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'n' (line 160)
        n_561622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'n', False)
        int_561623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'int')
        # Applying the binary operator '-' (line 160)
        result_sub_561624 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 37), '-', n_561622, int_561623)
        
        # Getting the type of 'x' (line 160)
        x_561625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 44), 'x', False)
        # Processing the call keyword arguments (line 160)
        kwargs_561626 = {}
        # Getting the type of 'spherical_in' (line 160)
        spherical_in_561621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 160)
        spherical_in_call_result_561627 = invoke(stypy.reporting.localization.Localization(__file__, 160, 24), spherical_in_561621, *[result_sub_561624, x_561625], **kwargs_561626)
        
        
        # Call to spherical_in(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'n' (line 160)
        n_561629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 62), 'n', False)
        int_561630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 66), 'int')
        # Applying the binary operator '+' (line 160)
        result_add_561631 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 62), '+', n_561629, int_561630)
        
        # Getting the type of 'x' (line 160)
        x_561632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 68), 'x', False)
        # Processing the call keyword arguments (line 160)
        kwargs_561633 = {}
        # Getting the type of 'spherical_in' (line 160)
        spherical_in_561628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 160)
        spherical_in_call_result_561634 = invoke(stypy.reporting.localization.Localization(__file__, 160, 49), spherical_in_561628, *[result_add_561631, x_561632], **kwargs_561633)
        
        # Applying the binary operator '-' (line 160)
        result_sub_561635 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 24), '-', spherical_in_call_result_561627, spherical_in_call_result_561634)
        
        int_561636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 25), 'int')
        # Getting the type of 'n' (line 161)
        n_561637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'n', False)
        # Applying the binary operator '*' (line 161)
        result_mul_561638 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 25), '*', int_561636, n_561637)
        
        int_561639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'int')
        # Applying the binary operator '+' (line 161)
        result_add_561640 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 25), '+', result_mul_561638, int_561639)
        
        # Getting the type of 'x' (line 161)
        x_561641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'x', False)
        # Applying the binary operator 'div' (line 161)
        result_div_561642 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), 'div', result_add_561640, x_561641)
        
        
        # Call to spherical_in(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'n' (line 161)
        n_561644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), 'n', False)
        # Getting the type of 'x' (line 161)
        x_561645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 52), 'x', False)
        # Processing the call keyword arguments (line 161)
        kwargs_561646 = {}
        # Getting the type of 'spherical_in' (line 161)
        spherical_in_561643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 161)
        spherical_in_call_result_561647 = invoke(stypy.reporting.localization.Localization(__file__, 161, 36), spherical_in_561643, *[n_561644, x_561645], **kwargs_561646)
        
        # Applying the binary operator '*' (line 161)
        result_mul_561648 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 35), '*', result_div_561642, spherical_in_call_result_561647)
        
        # Processing the call keyword arguments (line 160)
        kwargs_561649 = {}
        # Getting the type of 'assert_allclose' (line 160)
        assert_allclose_561620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 160)
        assert_allclose_call_result_561650 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assert_allclose_561620, *[result_sub_561635, result_mul_561648], **kwargs_561649)
        
        
        # ################# End of 'test_spherical_in_recurrence_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_recurrence_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_561651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_recurrence_complex'
        return stypy_return_type_561651


    @norecursion
    def test_spherical_in_inf_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_inf_real'
        module_type_store = module_type_store.open_function_context('test_spherical_in_inf_real', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_inf_real')
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_inf_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_inf_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_inf_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_inf_real(...)' code ##################

        
        # Assigning a Num to a Name (line 165):
        
        # Assigning a Num to a Name (line 165):
        int_561652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 12), 'int')
        # Assigning a type to the variable 'n' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'n', int_561652)
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to array(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Obtaining an instance of the builtin type 'list' (line 166)
        list_561655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 166)
        # Adding element type (line 166)
        
        # Getting the type of 'inf' (line 166)
        inf_561656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 166)
        result___neg___561657 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 22), 'usub', inf_561656)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_561655, result___neg___561657)
        # Adding element type (line 166)
        # Getting the type of 'inf' (line 166)
        inf_561658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 21), list_561655, inf_561658)
        
        # Processing the call keyword arguments (line 166)
        kwargs_561659 = {}
        # Getting the type of 'np' (line 166)
        np_561653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 166)
        array_561654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), np_561653, 'array')
        # Calling array(args, kwargs) (line 166)
        array_call_result_561660 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), array_561654, *[list_561655], **kwargs_561659)
        
        # Assigning a type to the variable 'x' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'x', array_call_result_561660)
        
        # Call to assert_allclose(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to spherical_in(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'n' (line 167)
        n_561663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'n', False)
        # Getting the type of 'x' (line 167)
        x_561664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 40), 'x', False)
        # Processing the call keyword arguments (line 167)
        kwargs_561665 = {}
        # Getting the type of 'spherical_in' (line 167)
        spherical_in_561662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 167)
        spherical_in_call_result_561666 = invoke(stypy.reporting.localization.Localization(__file__, 167, 24), spherical_in_561662, *[n_561663, x_561664], **kwargs_561665)
        
        
        # Call to array(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Obtaining an instance of the builtin type 'list' (line 167)
        list_561669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 167)
        # Adding element type (line 167)
        
        # Getting the type of 'inf' (line 167)
        inf_561670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 55), 'inf', False)
        # Applying the 'usub' unary operator (line 167)
        result___neg___561671 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 54), 'usub', inf_561670)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 53), list_561669, result___neg___561671)
        # Adding element type (line 167)
        # Getting the type of 'inf' (line 167)
        inf_561672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 60), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 53), list_561669, inf_561672)
        
        # Processing the call keyword arguments (line 167)
        kwargs_561673 = {}
        # Getting the type of 'np' (line 167)
        np_561667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 167)
        array_561668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 44), np_561667, 'array')
        # Calling array(args, kwargs) (line 167)
        array_call_result_561674 = invoke(stypy.reporting.localization.Localization(__file__, 167, 44), array_561668, *[list_561669], **kwargs_561673)
        
        # Processing the call keyword arguments (line 167)
        kwargs_561675 = {}
        # Getting the type of 'assert_allclose' (line 167)
        assert_allclose_561661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 167)
        assert_allclose_call_result_561676 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_allclose_561661, *[spherical_in_call_result_561666, array_call_result_561674], **kwargs_561675)
        
        
        # ################# End of 'test_spherical_in_inf_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_inf_real' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_561677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561677)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_inf_real'
        return stypy_return_type_561677


    @norecursion
    def test_spherical_in_inf_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_inf_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_in_inf_complex', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_inf_complex')
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_inf_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_inf_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_inf_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_inf_complex(...)' code ##################

        
        # Assigning a Num to a Name (line 176):
        
        # Assigning a Num to a Name (line 176):
        int_561678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 12), 'int')
        # Assigning a type to the variable 'n' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'n', int_561678)
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to array(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_561681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        
        # Getting the type of 'inf' (line 177)
        inf_561682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 177)
        result___neg___561683 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 22), 'usub', inf_561682)
        
        complex_561684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'complex')
        # Applying the binary operator '+' (line 177)
        result_add_561685 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 22), '+', result___neg___561683, complex_561684)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_561681, result_add_561685)
        # Adding element type (line 177)
        # Getting the type of 'inf' (line 177)
        inf_561686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 33), 'inf', False)
        complex_561687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 39), 'complex')
        # Applying the binary operator '+' (line 177)
        result_add_561688 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 33), '+', inf_561686, complex_561687)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_561681, result_add_561688)
        # Adding element type (line 177)
        # Getting the type of 'inf' (line 177)
        inf_561689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 43), 'inf', False)
        int_561690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 48), 'int')
        complex_561691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 50), 'complex')
        # Applying the binary operator '+' (line 177)
        result_add_561692 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 48), '+', int_561690, complex_561691)
        
        # Applying the binary operator '*' (line 177)
        result_mul_561693 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 43), '*', inf_561689, result_add_561692)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 21), list_561681, result_mul_561693)
        
        # Processing the call keyword arguments (line 177)
        kwargs_561694 = {}
        # Getting the type of 'np' (line 177)
        np_561679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 177)
        array_561680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), np_561679, 'array')
        # Calling array(args, kwargs) (line 177)
        array_call_result_561695 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), array_561680, *[list_561681], **kwargs_561694)
        
        # Assigning a type to the variable 'x' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'x', array_call_result_561695)
        
        # Call to assert_allclose(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to spherical_in(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'n' (line 178)
        n_561698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 37), 'n', False)
        # Getting the type of 'x' (line 178)
        x_561699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 40), 'x', False)
        # Processing the call keyword arguments (line 178)
        kwargs_561700 = {}
        # Getting the type of 'spherical_in' (line 178)
        spherical_in_561697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 178)
        spherical_in_call_result_561701 = invoke(stypy.reporting.localization.Localization(__file__, 178, 24), spherical_in_561697, *[n_561698, x_561699], **kwargs_561700)
        
        
        # Call to array(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_561704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        
        # Getting the type of 'inf' (line 178)
        inf_561705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 55), 'inf', False)
        # Applying the 'usub' unary operator (line 178)
        result___neg___561706 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 54), 'usub', inf_561705)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 53), list_561704, result___neg___561706)
        # Adding element type (line 178)
        # Getting the type of 'inf' (line 178)
        inf_561707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 60), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 53), list_561704, inf_561707)
        # Adding element type (line 178)
        # Getting the type of 'nan' (line 178)
        nan_561708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 65), 'nan', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 53), list_561704, nan_561708)
        
        # Processing the call keyword arguments (line 178)
        kwargs_561709 = {}
        # Getting the type of 'np' (line 178)
        np_561702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 178)
        array_561703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 44), np_561702, 'array')
        # Calling array(args, kwargs) (line 178)
        array_call_result_561710 = invoke(stypy.reporting.localization.Localization(__file__, 178, 44), array_561703, *[list_561704], **kwargs_561709)
        
        # Processing the call keyword arguments (line 178)
        kwargs_561711 = {}
        # Getting the type of 'assert_allclose' (line 178)
        assert_allclose_561696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 178)
        assert_allclose_call_result_561712 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assert_allclose_561696, *[spherical_in_call_result_561701, array_call_result_561710], **kwargs_561711)
        
        
        # ################# End of 'test_spherical_in_inf_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_inf_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_561713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561713)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_inf_complex'
        return stypy_return_type_561713


    @norecursion
    def test_spherical_in_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_at_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_in_at_zero', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalIn.test_spherical_in_at_zero')
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalIn.test_spherical_in_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.test_spherical_in_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to array(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_561716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        int_561717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561717)
        # Adding element type (line 183)
        int_561718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561718)
        # Adding element type (line 183)
        int_561719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561719)
        # Adding element type (line 183)
        int_561720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561720)
        # Adding element type (line 183)
        int_561721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561721)
        # Adding element type (line 183)
        int_561722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 21), list_561716, int_561722)
        
        # Processing the call keyword arguments (line 183)
        kwargs_561723 = {}
        # Getting the type of 'np' (line 183)
        np_561714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 183)
        array_561715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), np_561714, 'array')
        # Calling array(args, kwargs) (line 183)
        array_call_result_561724 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), array_561715, *[list_561716], **kwargs_561723)
        
        # Assigning a type to the variable 'n' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'n', array_call_result_561724)
        
        # Assigning a Num to a Name (line 184):
        
        # Assigning a Num to a Name (line 184):
        int_561725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 12), 'int')
        # Assigning a type to the variable 'x' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'x', int_561725)
        
        # Call to assert_allclose(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Call to spherical_in(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'n' (line 185)
        n_561728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'n', False)
        # Getting the type of 'x' (line 185)
        x_561729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 40), 'x', False)
        # Processing the call keyword arguments (line 185)
        kwargs_561730 = {}
        # Getting the type of 'spherical_in' (line 185)
        spherical_in_561727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 185)
        spherical_in_call_result_561731 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), spherical_in_561727, *[n_561728, x_561729], **kwargs_561730)
        
        
        # Call to array(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_561734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        # Adding element type (line 185)
        int_561735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561735)
        # Adding element type (line 185)
        int_561736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561736)
        # Adding element type (line 185)
        int_561737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561737)
        # Adding element type (line 185)
        int_561738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561738)
        # Adding element type (line 185)
        int_561739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561739)
        # Adding element type (line 185)
        int_561740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 53), list_561734, int_561740)
        
        # Processing the call keyword arguments (line 185)
        kwargs_561741 = {}
        # Getting the type of 'np' (line 185)
        np_561732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 185)
        array_561733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 44), np_561732, 'array')
        # Calling array(args, kwargs) (line 185)
        array_call_result_561742 = invoke(stypy.reporting.localization.Localization(__file__, 185, 44), array_561733, *[list_561734], **kwargs_561741)
        
        # Processing the call keyword arguments (line 185)
        kwargs_561743 = {}
        # Getting the type of 'assert_allclose' (line 185)
        assert_allclose_561726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 185)
        assert_allclose_call_result_561744 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_allclose_561726, *[spherical_in_call_result_561731, array_call_result_561742], **kwargs_561743)
        
        
        # ################# End of 'test_spherical_in_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_561745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_at_zero'
        return stypy_return_type_561745


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 0, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalIn.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalIn' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'TestSphericalIn', TestSphericalIn)
# Declaration of the 'TestSphericalKn' class

class TestSphericalKn:

    @norecursion
    def test_spherical_kn_exact(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_exact'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_exact', 189, 4, False)
        # Assigning a type to the variable 'self' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_exact')
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_exact.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_exact', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_exact', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_exact(...)' code ##################

        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to array(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_561748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        float_561749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), list_561748, float_561749)
        # Adding element type (line 191)
        float_561750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), list_561748, float_561750)
        # Adding element type (line 191)
        float_561751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), list_561748, float_561751)
        # Adding element type (line 191)
        float_561752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 21), list_561748, float_561752)
        
        # Processing the call keyword arguments (line 191)
        kwargs_561753 = {}
        # Getting the type of 'np' (line 191)
        np_561746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 191)
        array_561747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), np_561746, 'array')
        # Calling array(args, kwargs) (line 191)
        array_call_result_561754 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), array_561747, *[list_561748], **kwargs_561753)
        
        # Assigning a type to the variable 'x' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'x', array_call_result_561754)
        
        # Call to assert_allclose(...): (line 192)
        # Processing the call arguments (line 192)
        
        # Call to spherical_kn(...): (line 192)
        # Processing the call arguments (line 192)
        int_561757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 37), 'int')
        # Getting the type of 'x' (line 192)
        x_561758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 40), 'x', False)
        # Processing the call keyword arguments (line 192)
        kwargs_561759 = {}
        # Getting the type of 'spherical_kn' (line 192)
        spherical_kn_561756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 192)
        spherical_kn_call_result_561760 = invoke(stypy.reporting.localization.Localization(__file__, 192, 24), spherical_kn_561756, *[int_561757, x_561758], **kwargs_561759)
        
        # Getting the type of 'pi' (line 193)
        pi_561761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'pi', False)
        int_561762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 27), 'int')
        # Applying the binary operator 'div' (line 193)
        result_div_561763 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 24), 'div', pi_561761, int_561762)
        
        
        # Call to exp(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Getting the type of 'x' (line 193)
        x_561765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'x', False)
        # Applying the 'usub' unary operator (line 193)
        result___neg___561766 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 33), 'usub', x_561765)
        
        # Processing the call keyword arguments (line 193)
        kwargs_561767 = {}
        # Getting the type of 'exp' (line 193)
        exp_561764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'exp', False)
        # Calling exp(args, kwargs) (line 193)
        exp_call_result_561768 = invoke(stypy.reporting.localization.Localization(__file__, 193, 29), exp_561764, *[result___neg___561766], **kwargs_561767)
        
        # Applying the binary operator '*' (line 193)
        result_mul_561769 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 28), '*', result_div_561763, exp_call_result_561768)
        
        int_561770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 38), 'int')
        # Getting the type of 'x' (line 193)
        x_561771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 40), 'x', False)
        # Applying the binary operator 'div' (line 193)
        result_div_561772 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 38), 'div', int_561770, x_561771)
        
        int_561773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 44), 'int')
        # Getting the type of 'x' (line 193)
        x_561774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 46), 'x', False)
        int_561775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 49), 'int')
        # Applying the binary operator '**' (line 193)
        result_pow_561776 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 46), '**', x_561774, int_561775)
        
        # Applying the binary operator 'div' (line 193)
        result_div_561777 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 44), 'div', int_561773, result_pow_561776)
        
        # Applying the binary operator '+' (line 193)
        result_add_561778 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 38), '+', result_div_561772, result_div_561777)
        
        int_561779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 53), 'int')
        # Getting the type of 'x' (line 193)
        x_561780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 55), 'x', False)
        int_561781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 58), 'int')
        # Applying the binary operator '**' (line 193)
        result_pow_561782 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 55), '**', x_561780, int_561781)
        
        # Applying the binary operator 'div' (line 193)
        result_div_561783 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 53), 'div', int_561779, result_pow_561782)
        
        # Applying the binary operator '+' (line 193)
        result_add_561784 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 51), '+', result_add_561778, result_div_561783)
        
        # Applying the binary operator '*' (line 193)
        result_mul_561785 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 36), '*', result_mul_561769, result_add_561784)
        
        # Processing the call keyword arguments (line 192)
        kwargs_561786 = {}
        # Getting the type of 'assert_allclose' (line 192)
        assert_allclose_561755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 192)
        assert_allclose_call_result_561787 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), assert_allclose_561755, *[spherical_kn_call_result_561760, result_mul_561785], **kwargs_561786)
        
        
        # ################# End of 'test_spherical_kn_exact(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_exact' in the type store
        # Getting the type of 'stypy_return_type' (line 189)
        stypy_return_type_561788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561788)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_exact'
        return stypy_return_type_561788


    @norecursion
    def test_spherical_kn_recurrence_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_recurrence_real'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_recurrence_real', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_recurrence_real')
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_recurrence_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_recurrence_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_recurrence_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_recurrence_real(...)' code ##################

        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to array(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_561791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        int_561792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_561791, int_561792)
        # Adding element type (line 197)
        int_561793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_561791, int_561793)
        # Adding element type (line 197)
        int_561794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_561791, int_561794)
        # Adding element type (line 197)
        int_561795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_561791, int_561795)
        # Adding element type (line 197)
        int_561796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_561791, int_561796)
        
        # Processing the call keyword arguments (line 197)
        kwargs_561797 = {}
        # Getting the type of 'np' (line 197)
        np_561789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 197)
        array_561790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), np_561789, 'array')
        # Calling array(args, kwargs) (line 197)
        array_call_result_561798 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), array_561790, *[list_561791], **kwargs_561797)
        
        # Assigning a type to the variable 'n' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'n', array_call_result_561798)
        
        # Assigning a Num to a Name (line 198):
        
        # Assigning a Num to a Name (line 198):
        float_561799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 12), 'float')
        # Assigning a type to the variable 'x' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'x', float_561799)
        
        # Call to assert_allclose(...): (line 199)
        # Processing the call arguments (line 199)
        int_561801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 25), 'int')
        # Getting the type of 'n' (line 199)
        n_561802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'n', False)
        int_561803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 35), 'int')
        # Applying the binary operator '-' (line 199)
        result_sub_561804 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 31), '-', n_561802, int_561803)
        
        # Applying the binary operator '**' (line 199)
        result_pow_561805 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 24), '**', int_561801, result_sub_561804)
        
        
        # Call to spherical_kn(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'n' (line 199)
        n_561807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 51), 'n', False)
        int_561808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 55), 'int')
        # Applying the binary operator '-' (line 199)
        result_sub_561809 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 51), '-', n_561807, int_561808)
        
        # Getting the type of 'x' (line 199)
        x_561810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 58), 'x', False)
        # Processing the call keyword arguments (line 199)
        kwargs_561811 = {}
        # Getting the type of 'spherical_kn' (line 199)
        spherical_kn_561806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 199)
        spherical_kn_call_result_561812 = invoke(stypy.reporting.localization.Localization(__file__, 199, 38), spherical_kn_561806, *[result_sub_561809, x_561810], **kwargs_561811)
        
        # Applying the binary operator '*' (line 199)
        result_mul_561813 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 24), '*', result_pow_561805, spherical_kn_call_result_561812)
        
        int_561814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 64), 'int')
        # Getting the type of 'n' (line 199)
        n_561815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 70), 'n', False)
        int_561816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 74), 'int')
        # Applying the binary operator '+' (line 199)
        result_add_561817 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 70), '+', n_561815, int_561816)
        
        # Applying the binary operator '**' (line 199)
        result_pow_561818 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 63), '**', int_561814, result_add_561817)
        
        
        # Call to spherical_kn(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'n' (line 199)
        n_561820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 90), 'n', False)
        int_561821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 94), 'int')
        # Applying the binary operator '+' (line 199)
        result_add_561822 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 90), '+', n_561820, int_561821)
        
        # Getting the type of 'x' (line 199)
        x_561823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 96), 'x', False)
        # Processing the call keyword arguments (line 199)
        kwargs_561824 = {}
        # Getting the type of 'spherical_kn' (line 199)
        spherical_kn_561819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 77), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 199)
        spherical_kn_call_result_561825 = invoke(stypy.reporting.localization.Localization(__file__, 199, 77), spherical_kn_561819, *[result_add_561822, x_561823], **kwargs_561824)
        
        # Applying the binary operator '*' (line 199)
        result_mul_561826 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 63), '*', result_pow_561818, spherical_kn_call_result_561825)
        
        # Applying the binary operator '-' (line 199)
        result_sub_561827 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 24), '-', result_mul_561813, result_mul_561826)
        
        int_561828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 25), 'int')
        # Getting the type of 'n' (line 200)
        n_561829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'n', False)
        # Applying the binary operator '**' (line 200)
        result_pow_561830 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 24), '**', int_561828, n_561829)
        
        int_561831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 33), 'int')
        # Getting the type of 'n' (line 200)
        n_561832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'n', False)
        # Applying the binary operator '*' (line 200)
        result_mul_561833 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 33), '*', int_561831, n_561832)
        
        int_561834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'int')
        # Applying the binary operator '+' (line 200)
        result_add_561835 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 33), '+', result_mul_561833, int_561834)
        
        # Applying the binary operator '*' (line 200)
        result_mul_561836 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 24), '*', result_pow_561830, result_add_561835)
        
        # Getting the type of 'x' (line 200)
        x_561837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'x', False)
        # Applying the binary operator 'div' (line 200)
        result_div_561838 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 41), 'div', result_mul_561836, x_561837)
        
        
        # Call to spherical_kn(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'n' (line 200)
        n_561840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'n', False)
        # Getting the type of 'x' (line 200)
        x_561841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 60), 'x', False)
        # Processing the call keyword arguments (line 200)
        kwargs_561842 = {}
        # Getting the type of 'spherical_kn' (line 200)
        spherical_kn_561839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 44), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 200)
        spherical_kn_call_result_561843 = invoke(stypy.reporting.localization.Localization(__file__, 200, 44), spherical_kn_561839, *[n_561840, x_561841], **kwargs_561842)
        
        # Applying the binary operator '*' (line 200)
        result_mul_561844 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 43), '*', result_div_561838, spherical_kn_call_result_561843)
        
        # Processing the call keyword arguments (line 199)
        kwargs_561845 = {}
        # Getting the type of 'assert_allclose' (line 199)
        assert_allclose_561800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 199)
        assert_allclose_call_result_561846 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assert_allclose_561800, *[result_sub_561827, result_mul_561844], **kwargs_561845)
        
        
        # ################# End of 'test_spherical_kn_recurrence_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_recurrence_real' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_561847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_recurrence_real'
        return stypy_return_type_561847


    @norecursion
    def test_spherical_kn_recurrence_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_recurrence_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_recurrence_complex', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_recurrence_complex')
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_recurrence_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_recurrence_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_recurrence_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_recurrence_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to array(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_561850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        int_561851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_561850, int_561851)
        # Adding element type (line 204)
        int_561852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_561850, int_561852)
        # Adding element type (line 204)
        int_561853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_561850, int_561853)
        # Adding element type (line 204)
        int_561854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_561850, int_561854)
        # Adding element type (line 204)
        int_561855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 21), list_561850, int_561855)
        
        # Processing the call keyword arguments (line 204)
        kwargs_561856 = {}
        # Getting the type of 'np' (line 204)
        np_561848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 204)
        array_561849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), np_561848, 'array')
        # Calling array(args, kwargs) (line 204)
        array_call_result_561857 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), array_561849, *[list_561850], **kwargs_561856)
        
        # Assigning a type to the variable 'n' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'n', array_call_result_561857)
        
        # Assigning a BinOp to a Name (line 205):
        
        # Assigning a BinOp to a Name (line 205):
        float_561858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'float')
        complex_561859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 18), 'complex')
        # Applying the binary operator '+' (line 205)
        result_add_561860 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '+', float_561858, complex_561859)
        
        # Assigning a type to the variable 'x' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'x', result_add_561860)
        
        # Call to assert_allclose(...): (line 206)
        # Processing the call arguments (line 206)
        int_561862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 25), 'int')
        # Getting the type of 'n' (line 206)
        n_561863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'n', False)
        int_561864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 35), 'int')
        # Applying the binary operator '-' (line 206)
        result_sub_561865 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 31), '-', n_561863, int_561864)
        
        # Applying the binary operator '**' (line 206)
        result_pow_561866 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 24), '**', int_561862, result_sub_561865)
        
        
        # Call to spherical_kn(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'n' (line 206)
        n_561868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 51), 'n', False)
        int_561869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 55), 'int')
        # Applying the binary operator '-' (line 206)
        result_sub_561870 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 51), '-', n_561868, int_561869)
        
        # Getting the type of 'x' (line 206)
        x_561871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 58), 'x', False)
        # Processing the call keyword arguments (line 206)
        kwargs_561872 = {}
        # Getting the type of 'spherical_kn' (line 206)
        spherical_kn_561867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 38), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 206)
        spherical_kn_call_result_561873 = invoke(stypy.reporting.localization.Localization(__file__, 206, 38), spherical_kn_561867, *[result_sub_561870, x_561871], **kwargs_561872)
        
        # Applying the binary operator '*' (line 206)
        result_mul_561874 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 24), '*', result_pow_561866, spherical_kn_call_result_561873)
        
        int_561875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 64), 'int')
        # Getting the type of 'n' (line 206)
        n_561876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 70), 'n', False)
        int_561877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 74), 'int')
        # Applying the binary operator '+' (line 206)
        result_add_561878 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 70), '+', n_561876, int_561877)
        
        # Applying the binary operator '**' (line 206)
        result_pow_561879 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 63), '**', int_561875, result_add_561878)
        
        
        # Call to spherical_kn(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'n' (line 206)
        n_561881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 90), 'n', False)
        int_561882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 94), 'int')
        # Applying the binary operator '+' (line 206)
        result_add_561883 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 90), '+', n_561881, int_561882)
        
        # Getting the type of 'x' (line 206)
        x_561884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 96), 'x', False)
        # Processing the call keyword arguments (line 206)
        kwargs_561885 = {}
        # Getting the type of 'spherical_kn' (line 206)
        spherical_kn_561880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 77), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 206)
        spherical_kn_call_result_561886 = invoke(stypy.reporting.localization.Localization(__file__, 206, 77), spherical_kn_561880, *[result_add_561883, x_561884], **kwargs_561885)
        
        # Applying the binary operator '*' (line 206)
        result_mul_561887 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 63), '*', result_pow_561879, spherical_kn_call_result_561886)
        
        # Applying the binary operator '-' (line 206)
        result_sub_561888 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 24), '-', result_mul_561874, result_mul_561887)
        
        int_561889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 25), 'int')
        # Getting the type of 'n' (line 207)
        n_561890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 30), 'n', False)
        # Applying the binary operator '**' (line 207)
        result_pow_561891 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 24), '**', int_561889, n_561890)
        
        int_561892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'int')
        # Getting the type of 'n' (line 207)
        n_561893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 35), 'n', False)
        # Applying the binary operator '*' (line 207)
        result_mul_561894 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 33), '*', int_561892, n_561893)
        
        int_561895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 39), 'int')
        # Applying the binary operator '+' (line 207)
        result_add_561896 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 33), '+', result_mul_561894, int_561895)
        
        # Applying the binary operator '*' (line 207)
        result_mul_561897 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 24), '*', result_pow_561891, result_add_561896)
        
        # Getting the type of 'x' (line 207)
        x_561898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'x', False)
        # Applying the binary operator 'div' (line 207)
        result_div_561899 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 41), 'div', result_mul_561897, x_561898)
        
        
        # Call to spherical_kn(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'n' (line 207)
        n_561901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 57), 'n', False)
        # Getting the type of 'x' (line 207)
        x_561902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 60), 'x', False)
        # Processing the call keyword arguments (line 207)
        kwargs_561903 = {}
        # Getting the type of 'spherical_kn' (line 207)
        spherical_kn_561900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 207)
        spherical_kn_call_result_561904 = invoke(stypy.reporting.localization.Localization(__file__, 207, 44), spherical_kn_561900, *[n_561901, x_561902], **kwargs_561903)
        
        # Applying the binary operator '*' (line 207)
        result_mul_561905 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 43), '*', result_div_561899, spherical_kn_call_result_561904)
        
        # Processing the call keyword arguments (line 206)
        kwargs_561906 = {}
        # Getting the type of 'assert_allclose' (line 206)
        assert_allclose_561861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 206)
        assert_allclose_call_result_561907 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), assert_allclose_561861, *[result_sub_561888, result_mul_561905], **kwargs_561906)
        
        
        # ################# End of 'test_spherical_kn_recurrence_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_recurrence_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_561908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_recurrence_complex'
        return stypy_return_type_561908


    @norecursion
    def test_spherical_kn_inf_real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_inf_real'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_inf_real', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_inf_real')
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_inf_real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_inf_real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_inf_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_inf_real(...)' code ##################

        
        # Assigning a Num to a Name (line 211):
        
        # Assigning a Num to a Name (line 211):
        int_561909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 12), 'int')
        # Assigning a type to the variable 'n' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'n', int_561909)
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to array(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_561912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        # Adding element type (line 212)
        
        # Getting the type of 'inf' (line 212)
        inf_561913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 212)
        result___neg___561914 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 22), 'usub', inf_561913)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_561912, result___neg___561914)
        # Adding element type (line 212)
        # Getting the type of 'inf' (line 212)
        inf_561915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'inf', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 21), list_561912, inf_561915)
        
        # Processing the call keyword arguments (line 212)
        kwargs_561916 = {}
        # Getting the type of 'np' (line 212)
        np_561910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 212)
        array_561911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), np_561910, 'array')
        # Calling array(args, kwargs) (line 212)
        array_call_result_561917 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), array_561911, *[list_561912], **kwargs_561916)
        
        # Assigning a type to the variable 'x' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'x', array_call_result_561917)
        
        # Call to assert_allclose(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to spherical_kn(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'n' (line 213)
        n_561920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'n', False)
        # Getting the type of 'x' (line 213)
        x_561921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 'x', False)
        # Processing the call keyword arguments (line 213)
        kwargs_561922 = {}
        # Getting the type of 'spherical_kn' (line 213)
        spherical_kn_561919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 213)
        spherical_kn_call_result_561923 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), spherical_kn_561919, *[n_561920, x_561921], **kwargs_561922)
        
        
        # Call to array(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_561926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        
        # Getting the type of 'inf' (line 213)
        inf_561927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 55), 'inf', False)
        # Applying the 'usub' unary operator (line 213)
        result___neg___561928 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 54), 'usub', inf_561927)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 53), list_561926, result___neg___561928)
        # Adding element type (line 213)
        int_561929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 53), list_561926, int_561929)
        
        # Processing the call keyword arguments (line 213)
        kwargs_561930 = {}
        # Getting the type of 'np' (line 213)
        np_561924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 213)
        array_561925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 44), np_561924, 'array')
        # Calling array(args, kwargs) (line 213)
        array_call_result_561931 = invoke(stypy.reporting.localization.Localization(__file__, 213, 44), array_561925, *[list_561926], **kwargs_561930)
        
        # Processing the call keyword arguments (line 213)
        kwargs_561932 = {}
        # Getting the type of 'assert_allclose' (line 213)
        assert_allclose_561918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 213)
        assert_allclose_call_result_561933 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_allclose_561918, *[spherical_kn_call_result_561923, array_call_result_561931], **kwargs_561932)
        
        
        # ################# End of 'test_spherical_kn_inf_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_inf_real' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_561934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_inf_real'
        return stypy_return_type_561934


    @norecursion
    def test_spherical_kn_inf_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_inf_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_inf_complex', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_inf_complex')
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_inf_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_inf_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_inf_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_inf_complex(...)' code ##################

        
        # Assigning a Num to a Name (line 220):
        
        # Assigning a Num to a Name (line 220):
        int_561935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 12), 'int')
        # Assigning a type to the variable 'n' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'n', int_561935)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to array(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_561938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        # Adding element type (line 221)
        
        # Getting the type of 'inf' (line 221)
        inf_561939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 23), 'inf', False)
        # Applying the 'usub' unary operator (line 221)
        result___neg___561940 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 22), 'usub', inf_561939)
        
        complex_561941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 29), 'complex')
        # Applying the binary operator '+' (line 221)
        result_add_561942 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 22), '+', result___neg___561940, complex_561941)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_561938, result_add_561942)
        # Adding element type (line 221)
        # Getting the type of 'inf' (line 221)
        inf_561943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 33), 'inf', False)
        complex_561944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 39), 'complex')
        # Applying the binary operator '+' (line 221)
        result_add_561945 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 33), '+', inf_561943, complex_561944)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_561938, result_add_561945)
        # Adding element type (line 221)
        # Getting the type of 'inf' (line 221)
        inf_561946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), 'inf', False)
        int_561947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 48), 'int')
        complex_561948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 50), 'complex')
        # Applying the binary operator '+' (line 221)
        result_add_561949 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 48), '+', int_561947, complex_561948)
        
        # Applying the binary operator '*' (line 221)
        result_mul_561950 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 43), '*', inf_561946, result_add_561949)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 21), list_561938, result_mul_561950)
        
        # Processing the call keyword arguments (line 221)
        kwargs_561951 = {}
        # Getting the type of 'np' (line 221)
        np_561936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 221)
        array_561937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), np_561936, 'array')
        # Calling array(args, kwargs) (line 221)
        array_call_result_561952 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), array_561937, *[list_561938], **kwargs_561951)
        
        # Assigning a type to the variable 'x' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'x', array_call_result_561952)
        
        # Call to assert_allclose(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Call to spherical_kn(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'n' (line 222)
        n_561955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 37), 'n', False)
        # Getting the type of 'x' (line 222)
        x_561956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'x', False)
        # Processing the call keyword arguments (line 222)
        kwargs_561957 = {}
        # Getting the type of 'spherical_kn' (line 222)
        spherical_kn_561954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 222)
        spherical_kn_call_result_561958 = invoke(stypy.reporting.localization.Localization(__file__, 222, 24), spherical_kn_561954, *[n_561955, x_561956], **kwargs_561957)
        
        
        # Call to array(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_561961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        
        # Getting the type of 'inf' (line 222)
        inf_561962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 55), 'inf', False)
        # Applying the 'usub' unary operator (line 222)
        result___neg___561963 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 54), 'usub', inf_561962)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 53), list_561961, result___neg___561963)
        # Adding element type (line 222)
        int_561964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 53), list_561961, int_561964)
        # Adding element type (line 222)
        # Getting the type of 'nan' (line 222)
        nan_561965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 63), 'nan', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 53), list_561961, nan_561965)
        
        # Processing the call keyword arguments (line 222)
        kwargs_561966 = {}
        # Getting the type of 'np' (line 222)
        np_561959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 44), 'np', False)
        # Obtaining the member 'array' of a type (line 222)
        array_561960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 44), np_561959, 'array')
        # Calling array(args, kwargs) (line 222)
        array_call_result_561967 = invoke(stypy.reporting.localization.Localization(__file__, 222, 44), array_561960, *[list_561961], **kwargs_561966)
        
        # Processing the call keyword arguments (line 222)
        kwargs_561968 = {}
        # Getting the type of 'assert_allclose' (line 222)
        assert_allclose_561953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 222)
        assert_allclose_call_result_561969 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), assert_allclose_561953, *[spherical_kn_call_result_561958, array_call_result_561967], **kwargs_561968)
        
        
        # ################# End of 'test_spherical_kn_inf_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_inf_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_561970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_561970)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_inf_complex'
        return stypy_return_type_561970


    @norecursion
    def test_spherical_kn_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_at_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_at_zero', 224, 4, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_at_zero')
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to array(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_561973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        int_561974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561974)
        # Adding element type (line 226)
        int_561975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561975)
        # Adding element type (line 226)
        int_561976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561976)
        # Adding element type (line 226)
        int_561977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561977)
        # Adding element type (line 226)
        int_561978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561978)
        # Adding element type (line 226)
        int_561979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 21), list_561973, int_561979)
        
        # Processing the call keyword arguments (line 226)
        kwargs_561980 = {}
        # Getting the type of 'np' (line 226)
        np_561971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 226)
        array_561972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), np_561971, 'array')
        # Calling array(args, kwargs) (line 226)
        array_call_result_561981 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), array_561972, *[list_561973], **kwargs_561980)
        
        # Assigning a type to the variable 'n' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'n', array_call_result_561981)
        
        # Assigning a Num to a Name (line 227):
        
        # Assigning a Num to a Name (line 227):
        int_561982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 12), 'int')
        # Assigning a type to the variable 'x' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'x', int_561982)
        
        # Call to assert_allclose(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to spherical_kn(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'n' (line 228)
        n_561985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 37), 'n', False)
        # Getting the type of 'x' (line 228)
        x_561986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 40), 'x', False)
        # Processing the call keyword arguments (line 228)
        kwargs_561987 = {}
        # Getting the type of 'spherical_kn' (line 228)
        spherical_kn_561984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 228)
        spherical_kn_call_result_561988 = invoke(stypy.reporting.localization.Localization(__file__, 228, 24), spherical_kn_561984, *[n_561985, x_561986], **kwargs_561987)
        
        # Getting the type of 'inf' (line 228)
        inf_561989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 44), 'inf', False)
        
        # Call to ones(...): (line 228)
        # Processing the call keyword arguments (line 228)
        # Getting the type of 'n' (line 228)
        n_561992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 62), 'n', False)
        # Obtaining the member 'shape' of a type (line 228)
        shape_561993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 62), n_561992, 'shape')
        keyword_561994 = shape_561993
        kwargs_561995 = {'shape': keyword_561994}
        # Getting the type of 'np' (line 228)
        np_561990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'np', False)
        # Obtaining the member 'ones' of a type (line 228)
        ones_561991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 48), np_561990, 'ones')
        # Calling ones(args, kwargs) (line 228)
        ones_call_result_561996 = invoke(stypy.reporting.localization.Localization(__file__, 228, 48), ones_561991, *[], **kwargs_561995)
        
        # Applying the binary operator '*' (line 228)
        result_mul_561997 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 44), '*', inf_561989, ones_call_result_561996)
        
        # Processing the call keyword arguments (line 228)
        kwargs_561998 = {}
        # Getting the type of 'assert_allclose' (line 228)
        assert_allclose_561983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 228)
        assert_allclose_call_result_561999 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert_allclose_561983, *[spherical_kn_call_result_561988, result_mul_561997], **kwargs_561998)
        
        
        # ################# End of 'test_spherical_kn_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 224)
        stypy_return_type_562000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_at_zero'
        return stypy_return_type_562000


    @norecursion
    def test_spherical_kn_at_zero_complex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_kn_at_zero_complex'
        module_type_store = module_type_store.open_function_context('test_spherical_kn_at_zero_complex', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_function_name', 'TestSphericalKn.test_spherical_kn_at_zero_complex')
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKn.test_spherical_kn_at_zero_complex.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.test_spherical_kn_at_zero_complex', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_kn_at_zero_complex', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_kn_at_zero_complex(...)' code ##################

        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to array(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining an instance of the builtin type 'list' (line 232)
        list_562003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 232)
        # Adding element type (line 232)
        int_562004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562004)
        # Adding element type (line 232)
        int_562005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562005)
        # Adding element type (line 232)
        int_562006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562006)
        # Adding element type (line 232)
        int_562007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562007)
        # Adding element type (line 232)
        int_562008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562008)
        # Adding element type (line 232)
        int_562009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 21), list_562003, int_562009)
        
        # Processing the call keyword arguments (line 232)
        kwargs_562010 = {}
        # Getting the type of 'np' (line 232)
        np_562001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 232)
        array_562002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), np_562001, 'array')
        # Calling array(args, kwargs) (line 232)
        array_call_result_562011 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), array_562002, *[list_562003], **kwargs_562010)
        
        # Assigning a type to the variable 'n' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'n', array_call_result_562011)
        
        # Assigning a BinOp to a Name (line 233):
        
        # Assigning a BinOp to a Name (line 233):
        int_562012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 12), 'int')
        complex_562013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'complex')
        # Applying the binary operator '+' (line 233)
        result_add_562014 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 12), '+', int_562012, complex_562013)
        
        # Assigning a type to the variable 'x' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'x', result_add_562014)
        
        # Call to assert_allclose(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Call to spherical_kn(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'n' (line 234)
        n_562017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'n', False)
        # Getting the type of 'x' (line 234)
        x_562018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 40), 'x', False)
        # Processing the call keyword arguments (line 234)
        kwargs_562019 = {}
        # Getting the type of 'spherical_kn' (line 234)
        spherical_kn_562016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 234)
        spherical_kn_call_result_562020 = invoke(stypy.reporting.localization.Localization(__file__, 234, 24), spherical_kn_562016, *[n_562017, x_562018], **kwargs_562019)
        
        # Getting the type of 'nan' (line 234)
        nan_562021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 44), 'nan', False)
        
        # Call to ones(...): (line 234)
        # Processing the call keyword arguments (line 234)
        # Getting the type of 'n' (line 234)
        n_562024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 62), 'n', False)
        # Obtaining the member 'shape' of a type (line 234)
        shape_562025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 62), n_562024, 'shape')
        keyword_562026 = shape_562025
        kwargs_562027 = {'shape': keyword_562026}
        # Getting the type of 'np' (line 234)
        np_562022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 48), 'np', False)
        # Obtaining the member 'ones' of a type (line 234)
        ones_562023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 48), np_562022, 'ones')
        # Calling ones(args, kwargs) (line 234)
        ones_call_result_562028 = invoke(stypy.reporting.localization.Localization(__file__, 234, 48), ones_562023, *[], **kwargs_562027)
        
        # Applying the binary operator '*' (line 234)
        result_mul_562029 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 44), '*', nan_562021, ones_call_result_562028)
        
        # Processing the call keyword arguments (line 234)
        kwargs_562030 = {}
        # Getting the type of 'assert_allclose' (line 234)
        assert_allclose_562015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 234)
        assert_allclose_call_result_562031 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), assert_allclose_562015, *[spherical_kn_call_result_562020, result_mul_562029], **kwargs_562030)
        
        
        # ################# End of 'test_spherical_kn_at_zero_complex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_kn_at_zero_complex' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_562032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_kn_at_zero_complex'
        return stypy_return_type_562032


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 188, 0, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKn.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalKn' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'TestSphericalKn', TestSphericalKn)
# Declaration of the 'SphericalDerivativesTestCase' class

class SphericalDerivativesTestCase:

    @norecursion
    def fundamental_theorem(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fundamental_theorem'
        module_type_store = module_type_store.open_function_context('fundamental_theorem', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_localization', localization)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_type_store', module_type_store)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_function_name', 'SphericalDerivativesTestCase.fundamental_theorem')
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_param_names_list', ['n', 'a', 'b'])
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_varargs_param_name', None)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_call_defaults', defaults)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_call_varargs', varargs)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SphericalDerivativesTestCase.fundamental_theorem.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalDerivativesTestCase.fundamental_theorem', ['n', 'a', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fundamental_theorem', localization, ['n', 'a', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fundamental_theorem(...)' code ##################

        
        # Assigning a Call to a Tuple (line 239):
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_562033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to quad(...): (line 239)
        # Processing the call arguments (line 239)

        @norecursion
        def _stypy_temp_lambda_483(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_483'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_483', 239, 35, True)
            # Passed parameters checking function
            _stypy_temp_lambda_483.stypy_localization = localization
            _stypy_temp_lambda_483.stypy_type_of_self = None
            _stypy_temp_lambda_483.stypy_type_store = module_type_store
            _stypy_temp_lambda_483.stypy_function_name = '_stypy_temp_lambda_483'
            _stypy_temp_lambda_483.stypy_param_names_list = ['z']
            _stypy_temp_lambda_483.stypy_varargs_param_name = None
            _stypy_temp_lambda_483.stypy_kwargs_param_name = None
            _stypy_temp_lambda_483.stypy_call_defaults = defaults
            _stypy_temp_lambda_483.stypy_call_varargs = varargs
            _stypy_temp_lambda_483.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_483', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_483', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to df(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'n' (line 239)
            n_562037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 53), 'n', False)
            # Getting the type of 'z' (line 239)
            z_562038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 56), 'z', False)
            # Processing the call keyword arguments (line 239)
            kwargs_562039 = {}
            # Getting the type of 'self' (line 239)
            self_562035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'self', False)
            # Obtaining the member 'df' of a type (line 239)
            df_562036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 45), self_562035, 'df')
            # Calling df(args, kwargs) (line 239)
            df_call_result_562040 = invoke(stypy.reporting.localization.Localization(__file__, 239, 45), df_562036, *[n_562037, z_562038], **kwargs_562039)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'stypy_return_type', df_call_result_562040)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_483' in the type store
            # Getting the type of 'stypy_return_type' (line 239)
            stypy_return_type_562041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_562041)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_483'
            return stypy_return_type_562041

        # Assigning a type to the variable '_stypy_temp_lambda_483' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), '_stypy_temp_lambda_483', _stypy_temp_lambda_483)
        # Getting the type of '_stypy_temp_lambda_483' (line 239)
        _stypy_temp_lambda_483_562042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), '_stypy_temp_lambda_483')
        # Getting the type of 'a' (line 239)
        a_562043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 60), 'a', False)
        # Getting the type of 'b' (line 239)
        b_562044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 63), 'b', False)
        # Processing the call keyword arguments (line 239)
        kwargs_562045 = {}
        # Getting the type of 'quad' (line 239)
        quad_562034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'quad', False)
        # Calling quad(args, kwargs) (line 239)
        quad_call_result_562046 = invoke(stypy.reporting.localization.Localization(__file__, 239, 30), quad_562034, *[_stypy_temp_lambda_483_562042, a_562043, b_562044], **kwargs_562045)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___562047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), quad_call_result_562046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_562048 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___562047, int_562033)
        
        # Assigning a type to the variable 'tuple_var_assignment_560888' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_560888', subscript_call_result_562048)
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_562049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to quad(...): (line 239)
        # Processing the call arguments (line 239)

        @norecursion
        def _stypy_temp_lambda_484(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_484'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_484', 239, 35, True)
            # Passed parameters checking function
            _stypy_temp_lambda_484.stypy_localization = localization
            _stypy_temp_lambda_484.stypy_type_of_self = None
            _stypy_temp_lambda_484.stypy_type_store = module_type_store
            _stypy_temp_lambda_484.stypy_function_name = '_stypy_temp_lambda_484'
            _stypy_temp_lambda_484.stypy_param_names_list = ['z']
            _stypy_temp_lambda_484.stypy_varargs_param_name = None
            _stypy_temp_lambda_484.stypy_kwargs_param_name = None
            _stypy_temp_lambda_484.stypy_call_defaults = defaults
            _stypy_temp_lambda_484.stypy_call_varargs = varargs
            _stypy_temp_lambda_484.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_484', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_484', ['z'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to df(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'n' (line 239)
            n_562053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 53), 'n', False)
            # Getting the type of 'z' (line 239)
            z_562054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 56), 'z', False)
            # Processing the call keyword arguments (line 239)
            kwargs_562055 = {}
            # Getting the type of 'self' (line 239)
            self_562051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'self', False)
            # Obtaining the member 'df' of a type (line 239)
            df_562052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 45), self_562051, 'df')
            # Calling df(args, kwargs) (line 239)
            df_call_result_562056 = invoke(stypy.reporting.localization.Localization(__file__, 239, 45), df_562052, *[n_562053, z_562054], **kwargs_562055)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'stypy_return_type', df_call_result_562056)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_484' in the type store
            # Getting the type of 'stypy_return_type' (line 239)
            stypy_return_type_562057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_562057)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_484'
            return stypy_return_type_562057

        # Assigning a type to the variable '_stypy_temp_lambda_484' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), '_stypy_temp_lambda_484', _stypy_temp_lambda_484)
        # Getting the type of '_stypy_temp_lambda_484' (line 239)
        _stypy_temp_lambda_484_562058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 35), '_stypy_temp_lambda_484')
        # Getting the type of 'a' (line 239)
        a_562059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 60), 'a', False)
        # Getting the type of 'b' (line 239)
        b_562060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 63), 'b', False)
        # Processing the call keyword arguments (line 239)
        kwargs_562061 = {}
        # Getting the type of 'quad' (line 239)
        quad_562050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 30), 'quad', False)
        # Calling quad(args, kwargs) (line 239)
        quad_call_result_562062 = invoke(stypy.reporting.localization.Localization(__file__, 239, 30), quad_562050, *[_stypy_temp_lambda_484_562058, a_562059, b_562060], **kwargs_562061)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___562063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), quad_call_result_562062, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_562064 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___562063, int_562049)
        
        # Assigning a type to the variable 'tuple_var_assignment_560889' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_560889', subscript_call_result_562064)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_560888' (line 239)
        tuple_var_assignment_560888_562065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_560888')
        # Assigning a type to the variable 'integral' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'integral', tuple_var_assignment_560888_562065)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_560889' (line 239)
        tuple_var_assignment_560889_562066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_560889')
        # Assigning a type to the variable 'tolerance' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 18), 'tolerance', tuple_var_assignment_560889_562066)
        
        # Call to assert_allclose(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'integral' (line 240)
        integral_562068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'integral', False)
        
        # Call to f(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'n' (line 241)
        n_562071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'n', False)
        # Getting the type of 'b' (line 241)
        b_562072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'b', False)
        # Processing the call keyword arguments (line 241)
        kwargs_562073 = {}
        # Getting the type of 'self' (line 241)
        self_562069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'self', False)
        # Obtaining the member 'f' of a type (line 241)
        f_562070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), self_562069, 'f')
        # Calling f(args, kwargs) (line 241)
        f_call_result_562074 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), f_562070, *[n_562071, b_562072], **kwargs_562073)
        
        
        # Call to f(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'n' (line 241)
        n_562077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 46), 'n', False)
        # Getting the type of 'a' (line 241)
        a_562078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 49), 'a', False)
        # Processing the call keyword arguments (line 241)
        kwargs_562079 = {}
        # Getting the type of 'self' (line 241)
        self_562075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 39), 'self', False)
        # Obtaining the member 'f' of a type (line 241)
        f_562076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 39), self_562075, 'f')
        # Calling f(args, kwargs) (line 241)
        f_call_result_562080 = invoke(stypy.reporting.localization.Localization(__file__, 241, 39), f_562076, *[n_562077, a_562078], **kwargs_562079)
        
        # Applying the binary operator '-' (line 241)
        result_sub_562081 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 24), '-', f_call_result_562074, f_call_result_562080)
        
        # Processing the call keyword arguments (line 240)
        # Getting the type of 'tolerance' (line 242)
        tolerance_562082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'tolerance', False)
        keyword_562083 = tolerance_562082
        kwargs_562084 = {'atol': keyword_562083}
        # Getting the type of 'assert_allclose' (line 240)
        assert_allclose_562067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 240)
        assert_allclose_call_result_562085 = invoke(stypy.reporting.localization.Localization(__file__, 240, 8), assert_allclose_562067, *[integral_562068, result_sub_562081], **kwargs_562084)
        
        
        # ################# End of 'fundamental_theorem(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fundamental_theorem' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_562086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fundamental_theorem'
        return stypy_return_type_562086


    @norecursion
    def test_fundamental_theorem_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fundamental_theorem_0'
        module_type_store = module_type_store.open_function_context('test_fundamental_theorem_0', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_localization', localization)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_function_name', 'SphericalDerivativesTestCase.test_fundamental_theorem_0')
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_param_names_list', [])
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SphericalDerivativesTestCase.test_fundamental_theorem_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalDerivativesTestCase.test_fundamental_theorem_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fundamental_theorem_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fundamental_theorem_0(...)' code ##################

        
        # Call to fundamental_theorem(...): (line 246)
        # Processing the call arguments (line 246)
        int_562089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'int')
        float_562090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 36), 'float')
        float_562091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 41), 'float')
        # Processing the call keyword arguments (line 246)
        kwargs_562092 = {}
        # Getting the type of 'self' (line 246)
        self_562087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'fundamental_theorem' of a type (line 246)
        fundamental_theorem_562088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_562087, 'fundamental_theorem')
        # Calling fundamental_theorem(args, kwargs) (line 246)
        fundamental_theorem_call_result_562093 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), fundamental_theorem_562088, *[int_562089, float_562090, float_562091], **kwargs_562092)
        
        
        # ################# End of 'test_fundamental_theorem_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fundamental_theorem_0' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_562094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fundamental_theorem_0'
        return stypy_return_type_562094


    @norecursion
    def test_fundamental_theorem_7(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_fundamental_theorem_7'
        module_type_store = module_type_store.open_function_context('test_fundamental_theorem_7', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_localization', localization)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_type_store', module_type_store)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_function_name', 'SphericalDerivativesTestCase.test_fundamental_theorem_7')
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_param_names_list', [])
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_varargs_param_name', None)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_call_defaults', defaults)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_call_varargs', varargs)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SphericalDerivativesTestCase.test_fundamental_theorem_7.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalDerivativesTestCase.test_fundamental_theorem_7', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_fundamental_theorem_7', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_fundamental_theorem_7(...)' code ##################

        
        # Call to fundamental_theorem(...): (line 250)
        # Processing the call arguments (line 250)
        int_562097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 33), 'int')
        float_562098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 36), 'float')
        float_562099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 41), 'float')
        # Processing the call keyword arguments (line 250)
        kwargs_562100 = {}
        # Getting the type of 'self' (line 250)
        self_562095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'fundamental_theorem' of a type (line 250)
        fundamental_theorem_562096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_562095, 'fundamental_theorem')
        # Calling fundamental_theorem(args, kwargs) (line 250)
        fundamental_theorem_call_result_562101 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), fundamental_theorem_562096, *[int_562097, float_562098, float_562099], **kwargs_562100)
        
        
        # ################# End of 'test_fundamental_theorem_7(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_fundamental_theorem_7' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_562102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562102)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_fundamental_theorem_7'
        return stypy_return_type_562102


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 237, 0, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SphericalDerivativesTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SphericalDerivativesTestCase' (line 237)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 0), 'SphericalDerivativesTestCase', SphericalDerivativesTestCase)
# Declaration of the 'TestSphericalJnDerivatives' class
# Getting the type of 'SphericalDerivativesTestCase' (line 253)
SphericalDerivativesTestCase_562103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'SphericalDerivativesTestCase')

class TestSphericalJnDerivatives(SphericalDerivativesTestCase_562103, ):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_function_name', 'TestSphericalJnDerivatives.f')
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJnDerivatives.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnDerivatives.f', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to spherical_jn(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'n' (line 255)
        n_562105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'n', False)
        # Getting the type of 'z' (line 255)
        z_562106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 31), 'z', False)
        # Processing the call keyword arguments (line 255)
        kwargs_562107 = {}
        # Getting the type of 'spherical_jn' (line 255)
        spherical_jn_562104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 255)
        spherical_jn_call_result_562108 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), spherical_jn_562104, *[n_562105, z_562106], **kwargs_562107)
        
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', spherical_jn_call_result_562108)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_562109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_562109


    @norecursion
    def df(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'df'
        module_type_store = module_type_store.open_function_context('df', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_function_name', 'TestSphericalJnDerivatives.df')
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJnDerivatives.df.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnDerivatives.df', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'df', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'df(...)' code ##################

        
        # Call to spherical_jn(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'n' (line 258)
        n_562111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 28), 'n', False)
        # Getting the type of 'z' (line 258)
        z_562112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'z', False)
        # Processing the call keyword arguments (line 258)
        # Getting the type of 'True' (line 258)
        True_562113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 45), 'True', False)
        keyword_562114 = True_562113
        kwargs_562115 = {'derivative': keyword_562114}
        # Getting the type of 'spherical_jn' (line 258)
        spherical_jn_562110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 258)
        spherical_jn_call_result_562116 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), spherical_jn_562110, *[n_562111, z_562112], **kwargs_562115)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', spherical_jn_call_result_562116)
        
        # ################# End of 'df(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'df' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_562117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'df'
        return stypy_return_type_562117


    @norecursion
    def test_spherical_jn_d_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_jn_d_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_jn_d_zero', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalJnDerivatives.test_spherical_jn_d_zero')
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalJnDerivatives.test_spherical_jn_d_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnDerivatives.test_spherical_jn_d_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_jn_d_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_jn_d_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to array(...): (line 261)
        # Processing the call arguments (line 261)
        
        # Obtaining an instance of the builtin type 'list' (line 261)
        list_562120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 261)
        # Adding element type (line 261)
        int_562121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_562120, int_562121)
        # Adding element type (line 261)
        int_562122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_562120, int_562122)
        # Adding element type (line 261)
        int_562123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_562120, int_562123)
        # Adding element type (line 261)
        int_562124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_562120, int_562124)
        # Adding element type (line 261)
        int_562125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 261, 21), list_562120, int_562125)
        
        # Processing the call keyword arguments (line 261)
        kwargs_562126 = {}
        # Getting the type of 'np' (line 261)
        np_562118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 261)
        array_562119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), np_562118, 'array')
        # Calling array(args, kwargs) (line 261)
        array_call_result_562127 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), array_562119, *[list_562120], **kwargs_562126)
        
        # Assigning a type to the variable 'n' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'n', array_call_result_562127)
        
        # Call to assert_allclose(...): (line 262)
        # Processing the call arguments (line 262)
        
        # Call to spherical_jn(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'n' (line 262)
        n_562130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 37), 'n', False)
        int_562131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 40), 'int')
        # Processing the call keyword arguments (line 262)
        # Getting the type of 'True' (line 262)
        True_562132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 54), 'True', False)
        keyword_562133 = True_562132
        kwargs_562134 = {'derivative': keyword_562133}
        # Getting the type of 'spherical_jn' (line 262)
        spherical_jn_562129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 262)
        spherical_jn_call_result_562135 = invoke(stypy.reporting.localization.Localization(__file__, 262, 24), spherical_jn_562129, *[n_562130, int_562131], **kwargs_562134)
        
        
        # Call to zeros(...): (line 263)
        # Processing the call arguments (line 263)
        int_562138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'int')
        # Processing the call keyword arguments (line 263)
        kwargs_562139 = {}
        # Getting the type of 'np' (line 263)
        np_562136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'np', False)
        # Obtaining the member 'zeros' of a type (line 263)
        zeros_562137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), np_562136, 'zeros')
        # Calling zeros(args, kwargs) (line 263)
        zeros_call_result_562140 = invoke(stypy.reporting.localization.Localization(__file__, 263, 24), zeros_562137, *[int_562138], **kwargs_562139)
        
        # Processing the call keyword arguments (line 262)
        kwargs_562141 = {}
        # Getting the type of 'assert_allclose' (line 262)
        assert_allclose_562128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 262)
        assert_allclose_call_result_562142 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), assert_allclose_562128, *[spherical_jn_call_result_562135, zeros_call_result_562140], **kwargs_562141)
        
        
        # ################# End of 'test_spherical_jn_d_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_jn_d_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_562143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562143)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_jn_d_zero'
        return stypy_return_type_562143


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 253, 0, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalJnDerivatives.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalJnDerivatives' (line 253)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'TestSphericalJnDerivatives', TestSphericalJnDerivatives)
# Declaration of the 'TestSphericalYnDerivatives' class
# Getting the type of 'SphericalDerivativesTestCase' (line 266)
SphericalDerivativesTestCase_562144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'SphericalDerivativesTestCase')

class TestSphericalYnDerivatives(SphericalDerivativesTestCase_562144, ):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_function_name', 'TestSphericalYnDerivatives.f')
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYnDerivatives.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYnDerivatives.f', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to spherical_yn(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'n' (line 268)
        n_562146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'n', False)
        # Getting the type of 'z' (line 268)
        z_562147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 31), 'z', False)
        # Processing the call keyword arguments (line 268)
        kwargs_562148 = {}
        # Getting the type of 'spherical_yn' (line 268)
        spherical_yn_562145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 268)
        spherical_yn_call_result_562149 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), spherical_yn_562145, *[n_562146, z_562147], **kwargs_562148)
        
        # Assigning a type to the variable 'stypy_return_type' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'stypy_return_type', spherical_yn_call_result_562149)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_562150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_562150


    @norecursion
    def df(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'df'
        module_type_store = module_type_store.open_function_context('df', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_function_name', 'TestSphericalYnDerivatives.df')
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalYnDerivatives.df.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYnDerivatives.df', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'df', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'df(...)' code ##################

        
        # Call to spherical_yn(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'n' (line 271)
        n_562152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 28), 'n', False)
        # Getting the type of 'z' (line 271)
        z_562153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'z', False)
        # Processing the call keyword arguments (line 271)
        # Getting the type of 'True' (line 271)
        True_562154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 45), 'True', False)
        keyword_562155 = True_562154
        kwargs_562156 = {'derivative': keyword_562155}
        # Getting the type of 'spherical_yn' (line 271)
        spherical_yn_562151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 15), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 271)
        spherical_yn_call_result_562157 = invoke(stypy.reporting.localization.Localization(__file__, 271, 15), spherical_yn_562151, *[n_562152, z_562153], **kwargs_562156)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'stypy_return_type', spherical_yn_call_result_562157)
        
        # ################# End of 'df(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'df' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_562158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'df'
        return stypy_return_type_562158


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 266, 0, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalYnDerivatives.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalYnDerivatives' (line 266)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 0), 'TestSphericalYnDerivatives', TestSphericalYnDerivatives)
# Declaration of the 'TestSphericalInDerivatives' class
# Getting the type of 'SphericalDerivativesTestCase' (line 274)
SphericalDerivativesTestCase_562159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 33), 'SphericalDerivativesTestCase')

class TestSphericalInDerivatives(SphericalDerivativesTestCase_562159, ):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_function_name', 'TestSphericalInDerivatives.f')
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalInDerivatives.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalInDerivatives.f', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to spherical_in(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'n' (line 276)
        n_562161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 28), 'n', False)
        # Getting the type of 'z' (line 276)
        z_562162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'z', False)
        # Processing the call keyword arguments (line 276)
        kwargs_562163 = {}
        # Getting the type of 'spherical_in' (line 276)
        spherical_in_562160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 276)
        spherical_in_call_result_562164 = invoke(stypy.reporting.localization.Localization(__file__, 276, 15), spherical_in_562160, *[n_562161, z_562162], **kwargs_562163)
        
        # Assigning a type to the variable 'stypy_return_type' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'stypy_return_type', spherical_in_call_result_562164)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_562165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_562165


    @norecursion
    def df(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'df'
        module_type_store = module_type_store.open_function_context('df', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_function_name', 'TestSphericalInDerivatives.df')
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalInDerivatives.df.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalInDerivatives.df', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'df', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'df(...)' code ##################

        
        # Call to spherical_in(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'n' (line 279)
        n_562167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'n', False)
        # Getting the type of 'z' (line 279)
        z_562168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'z', False)
        # Processing the call keyword arguments (line 279)
        # Getting the type of 'True' (line 279)
        True_562169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 45), 'True', False)
        keyword_562170 = True_562169
        kwargs_562171 = {'derivative': keyword_562170}
        # Getting the type of 'spherical_in' (line 279)
        spherical_in_562166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 15), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 279)
        spherical_in_call_result_562172 = invoke(stypy.reporting.localization.Localization(__file__, 279, 15), spherical_in_562166, *[n_562167, z_562168], **kwargs_562171)
        
        # Assigning a type to the variable 'stypy_return_type' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'stypy_return_type', spherical_in_call_result_562172)
        
        # ################# End of 'df(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'df' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_562173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'df'
        return stypy_return_type_562173


    @norecursion
    def test_spherical_in_d_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_spherical_in_d_zero'
        module_type_store = module_type_store.open_function_context('test_spherical_in_d_zero', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_function_name', 'TestSphericalInDerivatives.test_spherical_in_d_zero')
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalInDerivatives.test_spherical_in_d_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalInDerivatives.test_spherical_in_d_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_spherical_in_d_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_spherical_in_d_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 282):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to array(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_562176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        int_562177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_562176, int_562177)
        # Adding element type (line 282)
        int_562178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_562176, int_562178)
        # Adding element type (line 282)
        int_562179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_562176, int_562179)
        # Adding element type (line 282)
        int_562180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_562176, int_562180)
        # Adding element type (line 282)
        int_562181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 21), list_562176, int_562181)
        
        # Processing the call keyword arguments (line 282)
        kwargs_562182 = {}
        # Getting the type of 'np' (line 282)
        np_562174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 282)
        array_562175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), np_562174, 'array')
        # Calling array(args, kwargs) (line 282)
        array_call_result_562183 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), array_562175, *[list_562176], **kwargs_562182)
        
        # Assigning a type to the variable 'n' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'n', array_call_result_562183)
        
        # Call to assert_allclose(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Call to spherical_in(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'n' (line 283)
        n_562186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'n', False)
        int_562187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 40), 'int')
        # Processing the call keyword arguments (line 283)
        # Getting the type of 'True' (line 283)
        True_562188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 54), 'True', False)
        keyword_562189 = True_562188
        kwargs_562190 = {'derivative': keyword_562189}
        # Getting the type of 'spherical_in' (line 283)
        spherical_in_562185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 283)
        spherical_in_call_result_562191 = invoke(stypy.reporting.localization.Localization(__file__, 283, 24), spherical_in_562185, *[n_562186, int_562187], **kwargs_562190)
        
        
        # Call to zeros(...): (line 284)
        # Processing the call arguments (line 284)
        int_562194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 33), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_562195 = {}
        # Getting the type of 'np' (line 284)
        np_562192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'np', False)
        # Obtaining the member 'zeros' of a type (line 284)
        zeros_562193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 24), np_562192, 'zeros')
        # Calling zeros(args, kwargs) (line 284)
        zeros_call_result_562196 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), zeros_562193, *[int_562194], **kwargs_562195)
        
        # Processing the call keyword arguments (line 283)
        kwargs_562197 = {}
        # Getting the type of 'assert_allclose' (line 283)
        assert_allclose_562184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 283)
        assert_allclose_call_result_562198 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert_allclose_562184, *[spherical_in_call_result_562191, zeros_call_result_562196], **kwargs_562197)
        
        
        # ################# End of 'test_spherical_in_d_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_spherical_in_d_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_562199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_spherical_in_d_zero'
        return stypy_return_type_562199


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 274, 0, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalInDerivatives.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalInDerivatives' (line 274)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'TestSphericalInDerivatives', TestSphericalInDerivatives)
# Declaration of the 'TestSphericalKnDerivatives' class
# Getting the type of 'SphericalDerivativesTestCase' (line 287)
SphericalDerivativesTestCase_562200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 33), 'SphericalDerivativesTestCase')

class TestSphericalKnDerivatives(SphericalDerivativesTestCase_562200, ):

    @norecursion
    def f(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_function_name', 'TestSphericalKnDerivatives.f')
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKnDerivatives.f.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKnDerivatives.f', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to spherical_kn(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'n' (line 289)
        n_562202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 28), 'n', False)
        # Getting the type of 'z' (line 289)
        z_562203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 31), 'z', False)
        # Processing the call keyword arguments (line 289)
        kwargs_562204 = {}
        # Getting the type of 'spherical_kn' (line 289)
        spherical_kn_562201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 289)
        spherical_kn_call_result_562205 = invoke(stypy.reporting.localization.Localization(__file__, 289, 15), spherical_kn_562201, *[n_562202, z_562203], **kwargs_562204)
        
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', spherical_kn_call_result_562205)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_562206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562206)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_562206


    @norecursion
    def df(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'df'
        module_type_store = module_type_store.open_function_context('df', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_function_name', 'TestSphericalKnDerivatives.df')
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_param_names_list', ['n', 'z'])
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalKnDerivatives.df.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKnDerivatives.df', ['n', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'df', localization, ['n', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'df(...)' code ##################

        
        # Call to spherical_kn(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'n' (line 292)
        n_562208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 28), 'n', False)
        # Getting the type of 'z' (line 292)
        z_562209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'z', False)
        # Processing the call keyword arguments (line 292)
        # Getting the type of 'True' (line 292)
        True_562210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 45), 'True', False)
        keyword_562211 = True_562210
        kwargs_562212 = {'derivative': keyword_562211}
        # Getting the type of 'spherical_kn' (line 292)
        spherical_kn_562207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 292)
        spherical_kn_call_result_562213 = invoke(stypy.reporting.localization.Localization(__file__, 292, 15), spherical_kn_562207, *[n_562208, z_562209], **kwargs_562212)
        
        # Assigning a type to the variable 'stypy_return_type' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'stypy_return_type', spherical_kn_call_result_562213)
        
        # ################# End of 'df(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'df' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_562214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'df'
        return stypy_return_type_562214


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 287, 0, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalKnDerivatives.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalKnDerivatives' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'TestSphericalKnDerivatives', TestSphericalKnDerivatives)
# Declaration of the 'TestSphericalOld' class

class TestSphericalOld:

    @norecursion
    def test_sph_in(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sph_in'
        module_type_store = module_type_store.open_function_context('test_sph_in', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_function_name', 'TestSphericalOld.test_sph_in')
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalOld.test_sph_in.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.test_sph_in', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sph_in', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sph_in(...)' code ##################

        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to empty(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Obtaining an instance of the builtin type 'tuple' (line 301)
        tuple_562217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 301)
        # Adding element type (line 301)
        int_562218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 24), tuple_562217, int_562218)
        # Adding element type (line 301)
        int_562219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 24), tuple_562217, int_562219)
        
        # Processing the call keyword arguments (line 301)
        kwargs_562220 = {}
        # Getting the type of 'np' (line 301)
        np_562215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 14), 'np', False)
        # Obtaining the member 'empty' of a type (line 301)
        empty_562216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 14), np_562215, 'empty')
        # Calling empty(args, kwargs) (line 301)
        empty_call_result_562221 = invoke(stypy.reporting.localization.Localization(__file__, 301, 14), empty_562216, *[tuple_562217], **kwargs_562220)
        
        # Assigning a type to the variable 'i1n' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'i1n', empty_call_result_562221)
        
        # Assigning a Num to a Name (line 302):
        
        # Assigning a Num to a Name (line 302):
        float_562222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 12), 'float')
        # Assigning a type to the variable 'x' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'x', float_562222)
        
        # Assigning a Call to a Subscript (line 304):
        
        # Assigning a Call to a Subscript (line 304):
        
        # Call to spherical_in(...): (line 304)
        # Processing the call arguments (line 304)
        int_562224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 33), 'int')
        # Getting the type of 'x' (line 304)
        x_562225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 36), 'x', False)
        # Processing the call keyword arguments (line 304)
        kwargs_562226 = {}
        # Getting the type of 'spherical_in' (line 304)
        spherical_in_562223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 304)
        spherical_in_call_result_562227 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), spherical_in_562223, *[int_562224, x_562225], **kwargs_562226)
        
        
        # Obtaining the type of the subscript
        int_562228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 12), 'int')
        # Getting the type of 'i1n' (line 304)
        i1n_562229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___562230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), i1n_562229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_562231 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), getitem___562230, int_562228)
        
        int_562232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 15), 'int')
        # Storing an element on a container (line 304)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 8), subscript_call_result_562231, (int_562232, spherical_in_call_result_562227))
        
        # Assigning a Call to a Subscript (line 305):
        
        # Assigning a Call to a Subscript (line 305):
        
        # Call to spherical_in(...): (line 305)
        # Processing the call arguments (line 305)
        int_562234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 33), 'int')
        # Getting the type of 'x' (line 305)
        x_562235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 36), 'x', False)
        # Processing the call keyword arguments (line 305)
        kwargs_562236 = {}
        # Getting the type of 'spherical_in' (line 305)
        spherical_in_562233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 305)
        spherical_in_call_result_562237 = invoke(stypy.reporting.localization.Localization(__file__, 305, 20), spherical_in_562233, *[int_562234, x_562235], **kwargs_562236)
        
        
        # Obtaining the type of the subscript
        int_562238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 12), 'int')
        # Getting the type of 'i1n' (line 305)
        i1n_562239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 305)
        getitem___562240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), i1n_562239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 305)
        subscript_call_result_562241 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), getitem___562240, int_562238)
        
        int_562242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 15), 'int')
        # Storing an element on a container (line 305)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 8), subscript_call_result_562241, (int_562242, spherical_in_call_result_562237))
        
        # Assigning a Call to a Subscript (line 306):
        
        # Assigning a Call to a Subscript (line 306):
        
        # Call to spherical_in(...): (line 306)
        # Processing the call arguments (line 306)
        int_562244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 33), 'int')
        # Getting the type of 'x' (line 306)
        x_562245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 36), 'x', False)
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'True' (line 306)
        True_562246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 50), 'True', False)
        keyword_562247 = True_562246
        kwargs_562248 = {'derivative': keyword_562247}
        # Getting the type of 'spherical_in' (line 306)
        spherical_in_562243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 306)
        spherical_in_call_result_562249 = invoke(stypy.reporting.localization.Localization(__file__, 306, 20), spherical_in_562243, *[int_562244, x_562245], **kwargs_562248)
        
        
        # Obtaining the type of the subscript
        int_562250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'int')
        # Getting the type of 'i1n' (line 306)
        i1n_562251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___562252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), i1n_562251, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_562253 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), getitem___562252, int_562250)
        
        int_562254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 15), 'int')
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 8), subscript_call_result_562253, (int_562254, spherical_in_call_result_562249))
        
        # Assigning a Call to a Subscript (line 307):
        
        # Assigning a Call to a Subscript (line 307):
        
        # Call to spherical_in(...): (line 307)
        # Processing the call arguments (line 307)
        int_562256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 33), 'int')
        # Getting the type of 'x' (line 307)
        x_562257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 36), 'x', False)
        # Processing the call keyword arguments (line 307)
        # Getting the type of 'True' (line 307)
        True_562258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 50), 'True', False)
        keyword_562259 = True_562258
        kwargs_562260 = {'derivative': keyword_562259}
        # Getting the type of 'spherical_in' (line 307)
        spherical_in_562255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 307)
        spherical_in_call_result_562261 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), spherical_in_562255, *[int_562256, x_562257], **kwargs_562260)
        
        
        # Obtaining the type of the subscript
        int_562262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 12), 'int')
        # Getting the type of 'i1n' (line 307)
        i1n_562263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___562264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), i1n_562263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_562265 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), getitem___562264, int_562262)
        
        int_562266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 15), 'int')
        # Storing an element on a container (line 307)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 8), subscript_call_result_562265, (int_562266, spherical_in_call_result_562261))
        
        # Assigning a Subscript to a Name (line 309):
        
        # Assigning a Subscript to a Name (line 309):
        
        # Obtaining the type of the subscript
        int_562267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 23), 'int')
        
        # Obtaining the type of the subscript
        int_562268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 20), 'int')
        # Getting the type of 'i1n' (line 309)
        i1n_562269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___562270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), i1n_562269, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_562271 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), getitem___562270, int_562268)
        
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___562272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), subscript_call_result_562271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_562273 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), getitem___562272, int_562267)
        
        # Assigning a type to the variable 'inp0' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'inp0', subscript_call_result_562273)
        
        # Assigning a BinOp to a Name (line 310):
        
        # Assigning a BinOp to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_562274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 23), 'int')
        
        # Obtaining the type of the subscript
        int_562275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 20), 'int')
        # Getting the type of 'i1n' (line 310)
        i1n_562276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___562277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), i1n_562276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_562278 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), getitem___562277, int_562275)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___562279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), subscript_call_result_562278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_562280 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), getitem___562279, int_562274)
        
        float_562281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 28), 'float')
        float_562282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 32), 'float')
        # Applying the binary operator 'div' (line 310)
        result_div_562283 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 28), 'div', float_562281, float_562282)
        
        
        # Obtaining the type of the subscript
        int_562284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 45), 'int')
        
        # Obtaining the type of the subscript
        int_562285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 42), 'int')
        # Getting the type of 'i1n' (line 310)
        i1n_562286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'i1n')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___562287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 38), i1n_562286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_562288 = invoke(stypy.reporting.localization.Localization(__file__, 310, 38), getitem___562287, int_562285)
        
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___562289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 38), subscript_call_result_562288, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_562290 = invoke(stypy.reporting.localization.Localization(__file__, 310, 38), getitem___562289, int_562284)
        
        # Applying the binary operator '*' (line 310)
        result_mul_562291 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 36), '*', result_div_562283, subscript_call_result_562290)
        
        # Applying the binary operator '-' (line 310)
        result_sub_562292 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 16), '-', subscript_call_result_562280, result_mul_562291)
        
        # Assigning a type to the variable 'inp1' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'inp1', result_sub_562292)
        
        # Call to assert_array_almost_equal(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining the type of the subscript
        int_562294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 38), 'int')
        # Getting the type of 'i1n' (line 311)
        i1n_562295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 34), 'i1n', False)
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___562296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 34), i1n_562295, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_562297 = invoke(stypy.reporting.localization.Localization(__file__, 311, 34), getitem___562296, int_562294)
        
        
        # Call to array(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_562300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        float_562301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 50), list_562300, float_562301)
        # Adding element type (line 311)
        float_562302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 50), list_562300, float_562302)
        
        # Processing the call keyword arguments (line 311)
        kwargs_562303 = {}
        # Getting the type of 'np' (line 311)
        np_562298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'np', False)
        # Obtaining the member 'array' of a type (line 311)
        array_562299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 41), np_562298, 'array')
        # Calling array(args, kwargs) (line 311)
        array_call_result_562304 = invoke(stypy.reporting.localization.Localization(__file__, 311, 41), array_562299, *[list_562300], **kwargs_562303)
        
        int_562305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 74), 'int')
        # Processing the call keyword arguments (line 311)
        kwargs_562306 = {}
        # Getting the type of 'assert_array_almost_equal' (line 311)
        assert_array_almost_equal_562293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 311)
        assert_array_almost_equal_call_result_562307 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), assert_array_almost_equal_562293, *[subscript_call_result_562297, array_call_result_562304, int_562305], **kwargs_562306)
        
        
        # Call to assert_array_almost_equal(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Obtaining the type of the subscript
        int_562309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 38), 'int')
        # Getting the type of 'i1n' (line 313)
        i1n_562310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'i1n', False)
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___562311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 34), i1n_562310, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_562312 = invoke(stypy.reporting.localization.Localization(__file__, 313, 34), getitem___562311, int_562309)
        
        
        # Obtaining an instance of the builtin type 'list' (line 313)
        list_562313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 313)
        # Adding element type (line 313)
        # Getting the type of 'inp0' (line 313)
        inp0_562314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'inp0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 41), list_562313, inp0_562314)
        # Adding element type (line 313)
        # Getting the type of 'inp1' (line 313)
        inp1_562315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 47), 'inp1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 41), list_562313, inp1_562315)
        
        int_562316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 53), 'int')
        # Processing the call keyword arguments (line 313)
        kwargs_562317 = {}
        # Getting the type of 'assert_array_almost_equal' (line 313)
        assert_array_almost_equal_562308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 313)
        assert_array_almost_equal_call_result_562318 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assert_array_almost_equal_562308, *[subscript_call_result_562312, list_562313, int_562316], **kwargs_562317)
        
        
        # ################# End of 'test_sph_in(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sph_in' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_562319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sph_in'
        return stypy_return_type_562319


    @norecursion
    def test_sph_in_kn_order0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sph_in_kn_order0'
        module_type_store = module_type_store.open_function_context('test_sph_in_kn_order0', 315, 4, False)
        # Assigning a type to the variable 'self' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_function_name', 'TestSphericalOld.test_sph_in_kn_order0')
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalOld.test_sph_in_kn_order0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.test_sph_in_kn_order0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sph_in_kn_order0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sph_in_kn_order0(...)' code ##################

        
        # Assigning a Num to a Name (line 316):
        
        # Assigning a Num to a Name (line 316):
        float_562320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 12), 'float')
        # Assigning a type to the variable 'x' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'x', float_562320)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to empty(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Obtaining an instance of the builtin type 'tuple' (line 317)
        tuple_562323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 317)
        # Adding element type (line 317)
        int_562324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 27), tuple_562323, int_562324)
        
        # Processing the call keyword arguments (line 317)
        kwargs_562325 = {}
        # Getting the type of 'np' (line 317)
        np_562321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 317)
        empty_562322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 17), np_562321, 'empty')
        # Calling empty(args, kwargs) (line 317)
        empty_call_result_562326 = invoke(stypy.reporting.localization.Localization(__file__, 317, 17), empty_562322, *[tuple_562323], **kwargs_562325)
        
        # Assigning a type to the variable 'sph_i0' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'sph_i0', empty_call_result_562326)
        
        # Assigning a Call to a Subscript (line 318):
        
        # Assigning a Call to a Subscript (line 318):
        
        # Call to spherical_in(...): (line 318)
        # Processing the call arguments (line 318)
        int_562328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 33), 'int')
        # Getting the type of 'x' (line 318)
        x_562329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 36), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_562330 = {}
        # Getting the type of 'spherical_in' (line 318)
        spherical_in_562327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 318)
        spherical_in_call_result_562331 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), spherical_in_562327, *[int_562328, x_562329], **kwargs_562330)
        
        # Getting the type of 'sph_i0' (line 318)
        sph_i0_562332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'sph_i0')
        int_562333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 15), 'int')
        # Storing an element on a container (line 318)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 8), sph_i0_562332, (int_562333, spherical_in_call_result_562331))
        
        # Assigning a Call to a Subscript (line 319):
        
        # Assigning a Call to a Subscript (line 319):
        
        # Call to spherical_in(...): (line 319)
        # Processing the call arguments (line 319)
        int_562335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 33), 'int')
        # Getting the type of 'x' (line 319)
        x_562336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 36), 'x', False)
        # Processing the call keyword arguments (line 319)
        # Getting the type of 'True' (line 319)
        True_562337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 50), 'True', False)
        keyword_562338 = True_562337
        kwargs_562339 = {'derivative': keyword_562338}
        # Getting the type of 'spherical_in' (line 319)
        spherical_in_562334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), 'spherical_in', False)
        # Calling spherical_in(args, kwargs) (line 319)
        spherical_in_call_result_562340 = invoke(stypy.reporting.localization.Localization(__file__, 319, 20), spherical_in_562334, *[int_562335, x_562336], **kwargs_562339)
        
        # Getting the type of 'sph_i0' (line 319)
        sph_i0_562341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'sph_i0')
        int_562342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 15), 'int')
        # Storing an element on a container (line 319)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 8), sph_i0_562341, (int_562342, spherical_in_call_result_562340))
        
        # Assigning a Call to a Name (line 320):
        
        # Assigning a Call to a Name (line 320):
        
        # Call to array(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining an instance of the builtin type 'list' (line 320)
        list_562345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 320)
        # Adding element type (line 320)
        
        # Call to sinh(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x' (line 320)
        x_562348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), 'x', False)
        # Processing the call keyword arguments (line 320)
        kwargs_562349 = {}
        # Getting the type of 'np' (line 320)
        np_562346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'np', False)
        # Obtaining the member 'sinh' of a type (line 320)
        sinh_562347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 36), np_562346, 'sinh')
        # Calling sinh(args, kwargs) (line 320)
        sinh_call_result_562350 = invoke(stypy.reporting.localization.Localization(__file__, 320, 36), sinh_562347, *[x_562348], **kwargs_562349)
        
        # Getting the type of 'x' (line 320)
        x_562351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 47), 'x', False)
        # Applying the binary operator 'div' (line 320)
        result_div_562352 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 36), 'div', sinh_call_result_562350, x_562351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 35), list_562345, result_div_562352)
        # Adding element type (line 320)
        
        # Call to cosh(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'x' (line 321)
        x_562355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 44), 'x', False)
        # Processing the call keyword arguments (line 321)
        kwargs_562356 = {}
        # Getting the type of 'np' (line 321)
        np_562353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 36), 'np', False)
        # Obtaining the member 'cosh' of a type (line 321)
        cosh_562354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 36), np_562353, 'cosh')
        # Calling cosh(args, kwargs) (line 321)
        cosh_call_result_562357 = invoke(stypy.reporting.localization.Localization(__file__, 321, 36), cosh_562354, *[x_562355], **kwargs_562356)
        
        # Getting the type of 'x' (line 321)
        x_562358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 47), 'x', False)
        # Applying the binary operator 'div' (line 321)
        result_div_562359 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 36), 'div', cosh_call_result_562357, x_562358)
        
        
        # Call to sinh(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'x' (line 321)
        x_562362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 57), 'x', False)
        # Processing the call keyword arguments (line 321)
        kwargs_562363 = {}
        # Getting the type of 'np' (line 321)
        np_562360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 49), 'np', False)
        # Obtaining the member 'sinh' of a type (line 321)
        sinh_562361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 49), np_562360, 'sinh')
        # Calling sinh(args, kwargs) (line 321)
        sinh_call_result_562364 = invoke(stypy.reporting.localization.Localization(__file__, 321, 49), sinh_562361, *[x_562362], **kwargs_562363)
        
        # Getting the type of 'x' (line 321)
        x_562365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 60), 'x', False)
        int_562366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 63), 'int')
        # Applying the binary operator '**' (line 321)
        result_pow_562367 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 60), '**', x_562365, int_562366)
        
        # Applying the binary operator 'div' (line 321)
        result_div_562368 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 49), 'div', sinh_call_result_562364, result_pow_562367)
        
        # Applying the binary operator '-' (line 321)
        result_sub_562369 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 36), '-', result_div_562359, result_div_562368)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 35), list_562345, result_sub_562369)
        
        # Processing the call keyword arguments (line 320)
        kwargs_562370 = {}
        # Getting the type of 'np' (line 320)
        np_562343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 320)
        array_562344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 26), np_562343, 'array')
        # Calling array(args, kwargs) (line 320)
        array_call_result_562371 = invoke(stypy.reporting.localization.Localization(__file__, 320, 26), array_562344, *[list_562345], **kwargs_562370)
        
        # Assigning a type to the variable 'sph_i0_expected' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'sph_i0_expected', array_call_result_562371)
        
        # Call to assert_array_almost_equal(...): (line 322)
        # Processing the call arguments (line 322)
        
        # Obtaining the type of the subscript
        # Getting the type of 'sph_i0' (line 322)
        sph_i0_562373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 37), 'sph_i0', False)
        # Getting the type of 'r_' (line 322)
        r__562374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 34), 'r_', False)
        # Obtaining the member '__getitem__' of a type (line 322)
        getitem___562375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 34), r__562374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 322)
        subscript_call_result_562376 = invoke(stypy.reporting.localization.Localization(__file__, 322, 34), getitem___562375, sph_i0_562373)
        
        # Getting the type of 'sph_i0_expected' (line 322)
        sph_i0_expected_562377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 46), 'sph_i0_expected', False)
        # Processing the call keyword arguments (line 322)
        kwargs_562378 = {}
        # Getting the type of 'assert_array_almost_equal' (line 322)
        assert_array_almost_equal_562372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 322)
        assert_array_almost_equal_call_result_562379 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), assert_array_almost_equal_562372, *[subscript_call_result_562376, sph_i0_expected_562377], **kwargs_562378)
        
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to empty(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Obtaining an instance of the builtin type 'tuple' (line 324)
        tuple_562382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 324)
        # Adding element type (line 324)
        int_562383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 27), tuple_562382, int_562383)
        
        # Processing the call keyword arguments (line 324)
        kwargs_562384 = {}
        # Getting the type of 'np' (line 324)
        np_562380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 324)
        empty_562381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 17), np_562380, 'empty')
        # Calling empty(args, kwargs) (line 324)
        empty_call_result_562385 = invoke(stypy.reporting.localization.Localization(__file__, 324, 17), empty_562381, *[tuple_562382], **kwargs_562384)
        
        # Assigning a type to the variable 'sph_k0' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'sph_k0', empty_call_result_562385)
        
        # Assigning a Call to a Subscript (line 325):
        
        # Assigning a Call to a Subscript (line 325):
        
        # Call to spherical_kn(...): (line 325)
        # Processing the call arguments (line 325)
        int_562387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 33), 'int')
        # Getting the type of 'x' (line 325)
        x_562388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 36), 'x', False)
        # Processing the call keyword arguments (line 325)
        kwargs_562389 = {}
        # Getting the type of 'spherical_kn' (line 325)
        spherical_kn_562386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 20), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 325)
        spherical_kn_call_result_562390 = invoke(stypy.reporting.localization.Localization(__file__, 325, 20), spherical_kn_562386, *[int_562387, x_562388], **kwargs_562389)
        
        # Getting the type of 'sph_k0' (line 325)
        sph_k0_562391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'sph_k0')
        int_562392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 15), 'int')
        # Storing an element on a container (line 325)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 8), sph_k0_562391, (int_562392, spherical_kn_call_result_562390))
        
        # Assigning a Call to a Subscript (line 326):
        
        # Assigning a Call to a Subscript (line 326):
        
        # Call to spherical_kn(...): (line 326)
        # Processing the call arguments (line 326)
        int_562394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 33), 'int')
        # Getting the type of 'x' (line 326)
        x_562395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'x', False)
        # Processing the call keyword arguments (line 326)
        # Getting the type of 'True' (line 326)
        True_562396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'True', False)
        keyword_562397 = True_562396
        kwargs_562398 = {'derivative': keyword_562397}
        # Getting the type of 'spherical_kn' (line 326)
        spherical_kn_562393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 20), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 326)
        spherical_kn_call_result_562399 = invoke(stypy.reporting.localization.Localization(__file__, 326, 20), spherical_kn_562393, *[int_562394, x_562395], **kwargs_562398)
        
        # Getting the type of 'sph_k0' (line 326)
        sph_k0_562400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'sph_k0')
        int_562401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 15), 'int')
        # Storing an element on a container (line 326)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 8), sph_k0_562400, (int_562401, spherical_kn_call_result_562399))
        
        # Assigning a Call to a Name (line 327):
        
        # Assigning a Call to a Name (line 327):
        
        # Call to array(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Obtaining an instance of the builtin type 'list' (line 327)
        list_562404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 327)
        # Adding element type (line 327)
        float_562405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 36), 'float')
        # Getting the type of 'pi' (line 327)
        pi_562406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 40), 'pi', False)
        # Applying the binary operator '*' (line 327)
        result_mul_562407 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 36), '*', float_562405, pi_562406)
        
        
        # Call to exp(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Getting the type of 'x' (line 327)
        x_562409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 48), 'x', False)
        # Applying the 'usub' unary operator (line 327)
        result___neg___562410 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 47), 'usub', x_562409)
        
        # Processing the call keyword arguments (line 327)
        kwargs_562411 = {}
        # Getting the type of 'exp' (line 327)
        exp_562408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 43), 'exp', False)
        # Calling exp(args, kwargs) (line 327)
        exp_call_result_562412 = invoke(stypy.reporting.localization.Localization(__file__, 327, 43), exp_562408, *[result___neg___562410], **kwargs_562411)
        
        # Applying the binary operator '*' (line 327)
        result_mul_562413 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 42), '*', result_mul_562407, exp_call_result_562412)
        
        # Getting the type of 'x' (line 327)
        x_562414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 51), 'x', False)
        # Applying the binary operator 'div' (line 327)
        result_div_562415 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 50), 'div', result_mul_562413, x_562414)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 35), list_562404, result_div_562415)
        # Adding element type (line 327)
        float_562416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 36), 'float')
        # Getting the type of 'pi' (line 328)
        pi_562417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 41), 'pi', False)
        # Applying the binary operator '*' (line 328)
        result_mul_562418 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 36), '*', float_562416, pi_562417)
        
        
        # Call to exp(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Getting the type of 'x' (line 328)
        x_562420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 49), 'x', False)
        # Applying the 'usub' unary operator (line 328)
        result___neg___562421 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 48), 'usub', x_562420)
        
        # Processing the call keyword arguments (line 328)
        kwargs_562422 = {}
        # Getting the type of 'exp' (line 328)
        exp_562419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 44), 'exp', False)
        # Calling exp(args, kwargs) (line 328)
        exp_call_result_562423 = invoke(stypy.reporting.localization.Localization(__file__, 328, 44), exp_562419, *[result___neg___562421], **kwargs_562422)
        
        # Applying the binary operator '*' (line 328)
        result_mul_562424 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 43), '*', result_mul_562418, exp_call_result_562423)
        
        int_562425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 53), 'int')
        # Getting the type of 'x' (line 328)
        x_562426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 55), 'x', False)
        # Applying the binary operator 'div' (line 328)
        result_div_562427 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 53), 'div', int_562425, x_562426)
        
        int_562428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 57), 'int')
        # Getting the type of 'x' (line 328)
        x_562429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 59), 'x', False)
        int_562430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 62), 'int')
        # Applying the binary operator '**' (line 328)
        result_pow_562431 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 59), '**', x_562429, int_562430)
        
        # Applying the binary operator 'div' (line 328)
        result_div_562432 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 57), 'div', int_562428, result_pow_562431)
        
        # Applying the binary operator '+' (line 328)
        result_add_562433 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 53), '+', result_div_562427, result_div_562432)
        
        # Applying the binary operator '*' (line 328)
        result_mul_562434 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 51), '*', result_mul_562424, result_add_562433)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 35), list_562404, result_mul_562434)
        
        # Processing the call keyword arguments (line 327)
        kwargs_562435 = {}
        # Getting the type of 'np' (line 327)
        np_562402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'np', False)
        # Obtaining the member 'array' of a type (line 327)
        array_562403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 26), np_562402, 'array')
        # Calling array(args, kwargs) (line 327)
        array_call_result_562436 = invoke(stypy.reporting.localization.Localization(__file__, 327, 26), array_562403, *[list_562404], **kwargs_562435)
        
        # Assigning a type to the variable 'sph_k0_expected' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'sph_k0_expected', array_call_result_562436)
        
        # Call to assert_array_almost_equal(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining the type of the subscript
        # Getting the type of 'sph_k0' (line 329)
        sph_k0_562438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 37), 'sph_k0', False)
        # Getting the type of 'r_' (line 329)
        r__562439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'r_', False)
        # Obtaining the member '__getitem__' of a type (line 329)
        getitem___562440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 34), r__562439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 329)
        subscript_call_result_562441 = invoke(stypy.reporting.localization.Localization(__file__, 329, 34), getitem___562440, sph_k0_562438)
        
        # Getting the type of 'sph_k0_expected' (line 329)
        sph_k0_expected_562442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 46), 'sph_k0_expected', False)
        # Processing the call keyword arguments (line 329)
        kwargs_562443 = {}
        # Getting the type of 'assert_array_almost_equal' (line 329)
        assert_array_almost_equal_562437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 329)
        assert_array_almost_equal_call_result_562444 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), assert_array_almost_equal_562437, *[subscript_call_result_562441, sph_k0_expected_562442], **kwargs_562443)
        
        
        # ################# End of 'test_sph_in_kn_order0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sph_in_kn_order0' in the type store
        # Getting the type of 'stypy_return_type' (line 315)
        stypy_return_type_562445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sph_in_kn_order0'
        return stypy_return_type_562445


    @norecursion
    def test_sph_jn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sph_jn'
        module_type_store = module_type_store.open_function_context('test_sph_jn', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_function_name', 'TestSphericalOld.test_sph_jn')
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalOld.test_sph_jn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.test_sph_jn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sph_jn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sph_jn(...)' code ##################

        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to empty(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_562448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        int_562449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 23), tuple_562448, int_562449)
        # Adding element type (line 332)
        int_562450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 23), tuple_562448, int_562450)
        
        # Processing the call keyword arguments (line 332)
        kwargs_562451 = {}
        # Getting the type of 'np' (line 332)
        np_562446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 332)
        empty_562447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 13), np_562446, 'empty')
        # Calling empty(args, kwargs) (line 332)
        empty_call_result_562452 = invoke(stypy.reporting.localization.Localization(__file__, 332, 13), empty_562447, *[tuple_562448], **kwargs_562451)
        
        # Assigning a type to the variable 's1' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 's1', empty_call_result_562452)
        
        # Assigning a Num to a Name (line 333):
        
        # Assigning a Num to a Name (line 333):
        float_562453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 12), 'float')
        # Assigning a type to the variable 'x' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'x', float_562453)
        
        # Assigning a Call to a Subscript (line 335):
        
        # Assigning a Call to a Subscript (line 335):
        
        # Call to spherical_jn(...): (line 335)
        # Processing the call arguments (line 335)
        int_562455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 32), 'int')
        # Getting the type of 'x' (line 335)
        x_562456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'x', False)
        # Processing the call keyword arguments (line 335)
        kwargs_562457 = {}
        # Getting the type of 'spherical_jn' (line 335)
        spherical_jn_562454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 335)
        spherical_jn_call_result_562458 = invoke(stypy.reporting.localization.Localization(__file__, 335, 19), spherical_jn_562454, *[int_562455, x_562456], **kwargs_562457)
        
        
        # Obtaining the type of the subscript
        int_562459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 11), 'int')
        # Getting the type of 's1' (line 335)
        s1_562460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___562461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 8), s1_562460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_562462 = invoke(stypy.reporting.localization.Localization(__file__, 335, 8), getitem___562461, int_562459)
        
        int_562463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 14), 'int')
        # Storing an element on a container (line 335)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 8), subscript_call_result_562462, (int_562463, spherical_jn_call_result_562458))
        
        # Assigning a Call to a Subscript (line 336):
        
        # Assigning a Call to a Subscript (line 336):
        
        # Call to spherical_jn(...): (line 336)
        # Processing the call arguments (line 336)
        int_562465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 32), 'int')
        # Getting the type of 'x' (line 336)
        x_562466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 35), 'x', False)
        # Processing the call keyword arguments (line 336)
        kwargs_562467 = {}
        # Getting the type of 'spherical_jn' (line 336)
        spherical_jn_562464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 336)
        spherical_jn_call_result_562468 = invoke(stypy.reporting.localization.Localization(__file__, 336, 19), spherical_jn_562464, *[int_562465, x_562466], **kwargs_562467)
        
        
        # Obtaining the type of the subscript
        int_562469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 11), 'int')
        # Getting the type of 's1' (line 336)
        s1_562470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___562471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 8), s1_562470, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_562472 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), getitem___562471, int_562469)
        
        int_562473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 14), 'int')
        # Storing an element on a container (line 336)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 8), subscript_call_result_562472, (int_562473, spherical_jn_call_result_562468))
        
        # Assigning a Call to a Subscript (line 337):
        
        # Assigning a Call to a Subscript (line 337):
        
        # Call to spherical_jn(...): (line 337)
        # Processing the call arguments (line 337)
        int_562475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 32), 'int')
        # Getting the type of 'x' (line 337)
        x_562476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 35), 'x', False)
        # Processing the call keyword arguments (line 337)
        kwargs_562477 = {}
        # Getting the type of 'spherical_jn' (line 337)
        spherical_jn_562474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 337)
        spherical_jn_call_result_562478 = invoke(stypy.reporting.localization.Localization(__file__, 337, 19), spherical_jn_562474, *[int_562475, x_562476], **kwargs_562477)
        
        
        # Obtaining the type of the subscript
        int_562479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 11), 'int')
        # Getting the type of 's1' (line 337)
        s1_562480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 337)
        getitem___562481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 8), s1_562480, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 337)
        subscript_call_result_562482 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), getitem___562481, int_562479)
        
        int_562483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 14), 'int')
        # Storing an element on a container (line 337)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 8), subscript_call_result_562482, (int_562483, spherical_jn_call_result_562478))
        
        # Assigning a Call to a Subscript (line 338):
        
        # Assigning a Call to a Subscript (line 338):
        
        # Call to spherical_jn(...): (line 338)
        # Processing the call arguments (line 338)
        int_562485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 32), 'int')
        # Getting the type of 'x' (line 338)
        x_562486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 35), 'x', False)
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'True' (line 338)
        True_562487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 49), 'True', False)
        keyword_562488 = True_562487
        kwargs_562489 = {'derivative': keyword_562488}
        # Getting the type of 'spherical_jn' (line 338)
        spherical_jn_562484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 338)
        spherical_jn_call_result_562490 = invoke(stypy.reporting.localization.Localization(__file__, 338, 19), spherical_jn_562484, *[int_562485, x_562486], **kwargs_562489)
        
        
        # Obtaining the type of the subscript
        int_562491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 11), 'int')
        # Getting the type of 's1' (line 338)
        s1_562492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 338)
        getitem___562493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 8), s1_562492, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 338)
        subscript_call_result_562494 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), getitem___562493, int_562491)
        
        int_562495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 14), 'int')
        # Storing an element on a container (line 338)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 338, 8), subscript_call_result_562494, (int_562495, spherical_jn_call_result_562490))
        
        # Assigning a Call to a Subscript (line 339):
        
        # Assigning a Call to a Subscript (line 339):
        
        # Call to spherical_jn(...): (line 339)
        # Processing the call arguments (line 339)
        int_562497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 32), 'int')
        # Getting the type of 'x' (line 339)
        x_562498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 35), 'x', False)
        # Processing the call keyword arguments (line 339)
        # Getting the type of 'True' (line 339)
        True_562499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 49), 'True', False)
        keyword_562500 = True_562499
        kwargs_562501 = {'derivative': keyword_562500}
        # Getting the type of 'spherical_jn' (line 339)
        spherical_jn_562496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 339)
        spherical_jn_call_result_562502 = invoke(stypy.reporting.localization.Localization(__file__, 339, 19), spherical_jn_562496, *[int_562497, x_562498], **kwargs_562501)
        
        
        # Obtaining the type of the subscript
        int_562503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 11), 'int')
        # Getting the type of 's1' (line 339)
        s1_562504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___562505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), s1_562504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_562506 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), getitem___562505, int_562503)
        
        int_562507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 14), 'int')
        # Storing an element on a container (line 339)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 8), subscript_call_result_562506, (int_562507, spherical_jn_call_result_562502))
        
        # Assigning a Call to a Subscript (line 340):
        
        # Assigning a Call to a Subscript (line 340):
        
        # Call to spherical_jn(...): (line 340)
        # Processing the call arguments (line 340)
        int_562509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 32), 'int')
        # Getting the type of 'x' (line 340)
        x_562510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 35), 'x', False)
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'True' (line 340)
        True_562511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 49), 'True', False)
        keyword_562512 = True_562511
        kwargs_562513 = {'derivative': keyword_562512}
        # Getting the type of 'spherical_jn' (line 340)
        spherical_jn_562508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 19), 'spherical_jn', False)
        # Calling spherical_jn(args, kwargs) (line 340)
        spherical_jn_call_result_562514 = invoke(stypy.reporting.localization.Localization(__file__, 340, 19), spherical_jn_562508, *[int_562509, x_562510], **kwargs_562513)
        
        
        # Obtaining the type of the subscript
        int_562515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 11), 'int')
        # Getting the type of 's1' (line 340)
        s1_562516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 's1')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___562517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), s1_562516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_562518 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), getitem___562517, int_562515)
        
        int_562519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 14), 'int')
        # Storing an element on a container (line 340)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 8), subscript_call_result_562518, (int_562519, spherical_jn_call_result_562514))
        
        # Assigning a UnaryOp to a Name (line 342):
        
        # Assigning a UnaryOp to a Name (line 342):
        
        
        # Obtaining the type of the subscript
        int_562520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 21), 'int')
        
        # Obtaining the type of the subscript
        int_562521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 18), 'int')
        # Getting the type of 's1' (line 342)
        s1_562522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 's1')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___562523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), s1_562522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_562524 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), getitem___562523, int_562521)
        
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___562525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), subscript_call_result_562524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_562526 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), getitem___562525, int_562520)
        
        # Applying the 'usub' unary operator (line 342)
        result___neg___562527 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 14), 'usub', subscript_call_result_562526)
        
        # Assigning a type to the variable 's10' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 's10', result___neg___562527)
        
        # Assigning a BinOp to a Name (line 343):
        
        # Assigning a BinOp to a Name (line 343):
        
        # Obtaining the type of the subscript
        int_562528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'int')
        
        # Obtaining the type of the subscript
        int_562529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 17), 'int')
        # Getting the type of 's1' (line 343)
        s1_562530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 's1')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___562531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 14), s1_562530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_562532 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), getitem___562531, int_562529)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___562533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 14), subscript_call_result_562532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_562534 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), getitem___562533, int_562528)
        
        float_562535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'float')
        float_562536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 27), 'float')
        # Applying the binary operator 'div' (line 343)
        result_div_562537 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 23), 'div', float_562535, float_562536)
        
        
        # Obtaining the type of the subscript
        int_562538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 37), 'int')
        
        # Obtaining the type of the subscript
        int_562539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'int')
        # Getting the type of 's1' (line 343)
        s1_562540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 's1')
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___562541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 31), s1_562540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_562542 = invoke(stypy.reporting.localization.Localization(__file__, 343, 31), getitem___562541, int_562539)
        
        # Obtaining the member '__getitem__' of a type (line 343)
        getitem___562543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 31), subscript_call_result_562542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 343)
        subscript_call_result_562544 = invoke(stypy.reporting.localization.Localization(__file__, 343, 31), getitem___562543, int_562538)
        
        # Applying the binary operator '*' (line 343)
        result_mul_562545 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 30), '*', result_div_562537, subscript_call_result_562544)
        
        # Applying the binary operator '-' (line 343)
        result_sub_562546 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 14), '-', subscript_call_result_562534, result_mul_562545)
        
        # Assigning a type to the variable 's11' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 's11', result_sub_562546)
        
        # Assigning a BinOp to a Name (line 344):
        
        # Assigning a BinOp to a Name (line 344):
        
        # Obtaining the type of the subscript
        int_562547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'int')
        
        # Obtaining the type of the subscript
        int_562548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 17), 'int')
        # Getting the type of 's1' (line 344)
        s1_562549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 14), 's1')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___562550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 14), s1_562549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_562551 = invoke(stypy.reporting.localization.Localization(__file__, 344, 14), getitem___562550, int_562548)
        
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___562552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 14), subscript_call_result_562551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_562553 = invoke(stypy.reporting.localization.Localization(__file__, 344, 14), getitem___562552, int_562547)
        
        float_562554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'float')
        float_562555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 27), 'float')
        # Applying the binary operator 'div' (line 344)
        result_div_562556 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 23), 'div', float_562554, float_562555)
        
        
        # Obtaining the type of the subscript
        int_562557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 37), 'int')
        
        # Obtaining the type of the subscript
        int_562558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 34), 'int')
        # Getting the type of 's1' (line 344)
        s1_562559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 's1')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___562560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 31), s1_562559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_562561 = invoke(stypy.reporting.localization.Localization(__file__, 344, 31), getitem___562560, int_562558)
        
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___562562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 31), subscript_call_result_562561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_562563 = invoke(stypy.reporting.localization.Localization(__file__, 344, 31), getitem___562562, int_562557)
        
        # Applying the binary operator '*' (line 344)
        result_mul_562564 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 30), '*', result_div_562556, subscript_call_result_562563)
        
        # Applying the binary operator '-' (line 344)
        result_sub_562565 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 14), '-', subscript_call_result_562553, result_mul_562564)
        
        # Assigning a type to the variable 's12' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 's12', result_sub_562565)
        
        # Call to assert_array_almost_equal(...): (line 345)
        # Processing the call arguments (line 345)
        
        # Obtaining the type of the subscript
        int_562567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 37), 'int')
        # Getting the type of 's1' (line 345)
        s1_562568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 34), 's1', False)
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___562569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 34), s1_562568, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_562570 = invoke(stypy.reporting.localization.Localization(__file__, 345, 34), getitem___562569, int_562567)
        
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_562571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        float_562572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 40), list_562571, float_562572)
        # Adding element type (line 345)
        float_562573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 40), list_562571, float_562573)
        # Adding element type (line 345)
        float_562574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 40), list_562571, float_562574)
        
        int_562575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 64), 'int')
        # Processing the call keyword arguments (line 345)
        kwargs_562576 = {}
        # Getting the type of 'assert_array_almost_equal' (line 345)
        assert_array_almost_equal_562566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 345)
        assert_array_almost_equal_call_result_562577 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), assert_array_almost_equal_562566, *[subscript_call_result_562570, list_562571, int_562575], **kwargs_562576)
        
        
        # Call to assert_array_almost_equal(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Obtaining the type of the subscript
        int_562579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 37), 'int')
        # Getting the type of 's1' (line 348)
        s1_562580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 's1', False)
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___562581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), s1_562580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_562582 = invoke(stypy.reporting.localization.Localization(__file__, 348, 34), getitem___562581, int_562579)
        
        
        # Obtaining an instance of the builtin type 'list' (line 348)
        list_562583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 348)
        # Adding element type (line 348)
        # Getting the type of 's10' (line 348)
        s10_562584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 's10', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 40), list_562583, s10_562584)
        # Adding element type (line 348)
        # Getting the type of 's11' (line 348)
        s11_562585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 45), 's11', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 40), list_562583, s11_562585)
        # Adding element type (line 348)
        # Getting the type of 's12' (line 348)
        s12_562586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 49), 's12', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 40), list_562583, s12_562586)
        
        int_562587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 54), 'int')
        # Processing the call keyword arguments (line 348)
        kwargs_562588 = {}
        # Getting the type of 'assert_array_almost_equal' (line 348)
        assert_array_almost_equal_562578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 348)
        assert_array_almost_equal_call_result_562589 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), assert_array_almost_equal_562578, *[subscript_call_result_562582, list_562583, int_562587], **kwargs_562588)
        
        
        # ################# End of 'test_sph_jn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sph_jn' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_562590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sph_jn'
        return stypy_return_type_562590


    @norecursion
    def test_sph_kn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sph_kn'
        module_type_store = module_type_store.open_function_context('test_sph_kn', 350, 4, False)
        # Assigning a type to the variable 'self' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_function_name', 'TestSphericalOld.test_sph_kn')
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalOld.test_sph_kn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.test_sph_kn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sph_kn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sph_kn(...)' code ##################

        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to empty(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Obtaining an instance of the builtin type 'tuple' (line 351)
        tuple_562593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 351)
        # Adding element type (line 351)
        int_562594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 23), tuple_562593, int_562594)
        # Adding element type (line 351)
        int_562595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 351, 23), tuple_562593, int_562595)
        
        # Processing the call keyword arguments (line 351)
        kwargs_562596 = {}
        # Getting the type of 'np' (line 351)
        np_562591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'np', False)
        # Obtaining the member 'empty' of a type (line 351)
        empty_562592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 13), np_562591, 'empty')
        # Calling empty(args, kwargs) (line 351)
        empty_call_result_562597 = invoke(stypy.reporting.localization.Localization(__file__, 351, 13), empty_562592, *[tuple_562593], **kwargs_562596)
        
        # Assigning a type to the variable 'kn' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'kn', empty_call_result_562597)
        
        # Assigning a Num to a Name (line 352):
        
        # Assigning a Num to a Name (line 352):
        float_562598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 12), 'float')
        # Assigning a type to the variable 'x' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'x', float_562598)
        
        # Assigning a Call to a Subscript (line 354):
        
        # Assigning a Call to a Subscript (line 354):
        
        # Call to spherical_kn(...): (line 354)
        # Processing the call arguments (line 354)
        int_562600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'int')
        # Getting the type of 'x' (line 354)
        x_562601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'x', False)
        # Processing the call keyword arguments (line 354)
        kwargs_562602 = {}
        # Getting the type of 'spherical_kn' (line 354)
        spherical_kn_562599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 354)
        spherical_kn_call_result_562603 = invoke(stypy.reporting.localization.Localization(__file__, 354, 19), spherical_kn_562599, *[int_562600, x_562601], **kwargs_562602)
        
        
        # Obtaining the type of the subscript
        int_562604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 11), 'int')
        # Getting the type of 'kn' (line 354)
        kn_562605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 354)
        getitem___562606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), kn_562605, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 354)
        subscript_call_result_562607 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___562606, int_562604)
        
        int_562608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 14), 'int')
        # Storing an element on a container (line 354)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), subscript_call_result_562607, (int_562608, spherical_kn_call_result_562603))
        
        # Assigning a Call to a Subscript (line 355):
        
        # Assigning a Call to a Subscript (line 355):
        
        # Call to spherical_kn(...): (line 355)
        # Processing the call arguments (line 355)
        int_562610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 32), 'int')
        # Getting the type of 'x' (line 355)
        x_562611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'x', False)
        # Processing the call keyword arguments (line 355)
        kwargs_562612 = {}
        # Getting the type of 'spherical_kn' (line 355)
        spherical_kn_562609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 355)
        spherical_kn_call_result_562613 = invoke(stypy.reporting.localization.Localization(__file__, 355, 19), spherical_kn_562609, *[int_562610, x_562611], **kwargs_562612)
        
        
        # Obtaining the type of the subscript
        int_562614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 11), 'int')
        # Getting the type of 'kn' (line 355)
        kn_562615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 355)
        getitem___562616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), kn_562615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 355)
        subscript_call_result_562617 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), getitem___562616, int_562614)
        
        int_562618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 14), 'int')
        # Storing an element on a container (line 355)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 8), subscript_call_result_562617, (int_562618, spherical_kn_call_result_562613))
        
        # Assigning a Call to a Subscript (line 356):
        
        # Assigning a Call to a Subscript (line 356):
        
        # Call to spherical_kn(...): (line 356)
        # Processing the call arguments (line 356)
        int_562620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 32), 'int')
        # Getting the type of 'x' (line 356)
        x_562621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'x', False)
        # Processing the call keyword arguments (line 356)
        kwargs_562622 = {}
        # Getting the type of 'spherical_kn' (line 356)
        spherical_kn_562619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 356)
        spherical_kn_call_result_562623 = invoke(stypy.reporting.localization.Localization(__file__, 356, 19), spherical_kn_562619, *[int_562620, x_562621], **kwargs_562622)
        
        
        # Obtaining the type of the subscript
        int_562624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 11), 'int')
        # Getting the type of 'kn' (line 356)
        kn_562625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___562626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), kn_562625, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_562627 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), getitem___562626, int_562624)
        
        int_562628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 14), 'int')
        # Storing an element on a container (line 356)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 8), subscript_call_result_562627, (int_562628, spherical_kn_call_result_562623))
        
        # Assigning a Call to a Subscript (line 357):
        
        # Assigning a Call to a Subscript (line 357):
        
        # Call to spherical_kn(...): (line 357)
        # Processing the call arguments (line 357)
        int_562630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 32), 'int')
        # Getting the type of 'x' (line 357)
        x_562631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'x', False)
        # Processing the call keyword arguments (line 357)
        # Getting the type of 'True' (line 357)
        True_562632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 49), 'True', False)
        keyword_562633 = True_562632
        kwargs_562634 = {'derivative': keyword_562633}
        # Getting the type of 'spherical_kn' (line 357)
        spherical_kn_562629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 357)
        spherical_kn_call_result_562635 = invoke(stypy.reporting.localization.Localization(__file__, 357, 19), spherical_kn_562629, *[int_562630, x_562631], **kwargs_562634)
        
        
        # Obtaining the type of the subscript
        int_562636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 11), 'int')
        # Getting the type of 'kn' (line 357)
        kn_562637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 357)
        getitem___562638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), kn_562637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 357)
        subscript_call_result_562639 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), getitem___562638, int_562636)
        
        int_562640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 14), 'int')
        # Storing an element on a container (line 357)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 357, 8), subscript_call_result_562639, (int_562640, spherical_kn_call_result_562635))
        
        # Assigning a Call to a Subscript (line 358):
        
        # Assigning a Call to a Subscript (line 358):
        
        # Call to spherical_kn(...): (line 358)
        # Processing the call arguments (line 358)
        int_562642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'int')
        # Getting the type of 'x' (line 358)
        x_562643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 35), 'x', False)
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'True' (line 358)
        True_562644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'True', False)
        keyword_562645 = True_562644
        kwargs_562646 = {'derivative': keyword_562645}
        # Getting the type of 'spherical_kn' (line 358)
        spherical_kn_562641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 358)
        spherical_kn_call_result_562647 = invoke(stypy.reporting.localization.Localization(__file__, 358, 19), spherical_kn_562641, *[int_562642, x_562643], **kwargs_562646)
        
        
        # Obtaining the type of the subscript
        int_562648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 11), 'int')
        # Getting the type of 'kn' (line 358)
        kn_562649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___562650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), kn_562649, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_562651 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___562650, int_562648)
        
        int_562652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 14), 'int')
        # Storing an element on a container (line 358)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 8), subscript_call_result_562651, (int_562652, spherical_kn_call_result_562647))
        
        # Assigning a Call to a Subscript (line 359):
        
        # Assigning a Call to a Subscript (line 359):
        
        # Call to spherical_kn(...): (line 359)
        # Processing the call arguments (line 359)
        int_562654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 32), 'int')
        # Getting the type of 'x' (line 359)
        x_562655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 35), 'x', False)
        # Processing the call keyword arguments (line 359)
        # Getting the type of 'True' (line 359)
        True_562656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 49), 'True', False)
        keyword_562657 = True_562656
        kwargs_562658 = {'derivative': keyword_562657}
        # Getting the type of 'spherical_kn' (line 359)
        spherical_kn_562653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'spherical_kn', False)
        # Calling spherical_kn(args, kwargs) (line 359)
        spherical_kn_call_result_562659 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), spherical_kn_562653, *[int_562654, x_562655], **kwargs_562658)
        
        
        # Obtaining the type of the subscript
        int_562660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 11), 'int')
        # Getting the type of 'kn' (line 359)
        kn_562661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'kn')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___562662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 8), kn_562661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_562663 = invoke(stypy.reporting.localization.Localization(__file__, 359, 8), getitem___562662, int_562660)
        
        int_562664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 14), 'int')
        # Storing an element on a container (line 359)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 8), subscript_call_result_562663, (int_562664, spherical_kn_call_result_562659))
        
        # Assigning a UnaryOp to a Name (line 361):
        
        # Assigning a UnaryOp to a Name (line 361):
        
        
        # Obtaining the type of the subscript
        int_562665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 21), 'int')
        
        # Obtaining the type of the subscript
        int_562666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 18), 'int')
        # Getting the type of 'kn' (line 361)
        kn_562667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 15), 'kn')
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___562668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 15), kn_562667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_562669 = invoke(stypy.reporting.localization.Localization(__file__, 361, 15), getitem___562668, int_562666)
        
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___562670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 15), subscript_call_result_562669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_562671 = invoke(stypy.reporting.localization.Localization(__file__, 361, 15), getitem___562670, int_562665)
        
        # Applying the 'usub' unary operator (line 361)
        result___neg___562672 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 14), 'usub', subscript_call_result_562671)
        
        # Assigning a type to the variable 'kn0' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'kn0', result___neg___562672)
        
        # Assigning a BinOp to a Name (line 362):
        
        # Assigning a BinOp to a Name (line 362):
        
        
        # Obtaining the type of the subscript
        int_562673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 21), 'int')
        
        # Obtaining the type of the subscript
        int_562674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 18), 'int')
        # Getting the type of 'kn' (line 362)
        kn_562675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'kn')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___562676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), kn_562675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_562677 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), getitem___562676, int_562674)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___562678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), subscript_call_result_562677, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_562679 = invoke(stypy.reporting.localization.Localization(__file__, 362, 15), getitem___562678, int_562673)
        
        # Applying the 'usub' unary operator (line 362)
        result___neg___562680 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 14), 'usub', subscript_call_result_562679)
        
        float_562681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 24), 'float')
        float_562682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'float')
        # Applying the binary operator 'div' (line 362)
        result_div_562683 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 24), 'div', float_562681, float_562682)
        
        
        # Obtaining the type of the subscript
        int_562684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 38), 'int')
        
        # Obtaining the type of the subscript
        int_562685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 35), 'int')
        # Getting the type of 'kn' (line 362)
        kn_562686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'kn')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___562687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), kn_562686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_562688 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___562687, int_562685)
        
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___562689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 32), subscript_call_result_562688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_562690 = invoke(stypy.reporting.localization.Localization(__file__, 362, 32), getitem___562689, int_562684)
        
        # Applying the binary operator '*' (line 362)
        result_mul_562691 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 31), '*', result_div_562683, subscript_call_result_562690)
        
        # Applying the binary operator '-' (line 362)
        result_sub_562692 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 14), '-', result___neg___562680, result_mul_562691)
        
        # Assigning a type to the variable 'kn1' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'kn1', result_sub_562692)
        
        # Assigning a BinOp to a Name (line 363):
        
        # Assigning a BinOp to a Name (line 363):
        
        
        # Obtaining the type of the subscript
        int_562693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 21), 'int')
        
        # Obtaining the type of the subscript
        int_562694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 18), 'int')
        # Getting the type of 'kn' (line 363)
        kn_562695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'kn')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___562696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), kn_562695, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_562697 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), getitem___562696, int_562694)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___562698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), subscript_call_result_562697, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_562699 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), getitem___562698, int_562693)
        
        # Applying the 'usub' unary operator (line 363)
        result___neg___562700 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 14), 'usub', subscript_call_result_562699)
        
        float_562701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 24), 'float')
        float_562702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 28), 'float')
        # Applying the binary operator 'div' (line 363)
        result_div_562703 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 24), 'div', float_562701, float_562702)
        
        
        # Obtaining the type of the subscript
        int_562704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 38), 'int')
        
        # Obtaining the type of the subscript
        int_562705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 35), 'int')
        # Getting the type of 'kn' (line 363)
        kn_562706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'kn')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___562707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 32), kn_562706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_562708 = invoke(stypy.reporting.localization.Localization(__file__, 363, 32), getitem___562707, int_562705)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___562709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 32), subscript_call_result_562708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_562710 = invoke(stypy.reporting.localization.Localization(__file__, 363, 32), getitem___562709, int_562704)
        
        # Applying the binary operator '*' (line 363)
        result_mul_562711 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 31), '*', result_div_562703, subscript_call_result_562710)
        
        # Applying the binary operator '-' (line 363)
        result_sub_562712 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 14), '-', result___neg___562700, result_mul_562711)
        
        # Assigning a type to the variable 'kn2' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'kn2', result_sub_562712)
        
        # Call to assert_array_almost_equal(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Obtaining the type of the subscript
        int_562714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 37), 'int')
        # Getting the type of 'kn' (line 364)
        kn_562715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 34), 'kn', False)
        # Obtaining the member '__getitem__' of a type (line 364)
        getitem___562716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 34), kn_562715, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 364)
        subscript_call_result_562717 = invoke(stypy.reporting.localization.Localization(__file__, 364, 34), getitem___562716, int_562714)
        
        
        # Obtaining an instance of the builtin type 'list' (line 364)
        list_562718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 364)
        # Adding element type (line 364)
        float_562719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 40), list_562718, float_562719)
        # Adding element type (line 364)
        float_562720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 40), list_562718, float_562720)
        # Adding element type (line 364)
        float_562721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 40), list_562718, float_562721)
        
        int_562722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 64), 'int')
        # Processing the call keyword arguments (line 364)
        kwargs_562723 = {}
        # Getting the type of 'assert_array_almost_equal' (line 364)
        assert_array_almost_equal_562713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 364)
        assert_array_almost_equal_call_result_562724 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), assert_array_almost_equal_562713, *[subscript_call_result_562717, list_562718, int_562722], **kwargs_562723)
        
        
        # Call to assert_array_almost_equal(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Obtaining the type of the subscript
        int_562726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 37), 'int')
        # Getting the type of 'kn' (line 367)
        kn_562727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'kn', False)
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___562728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 34), kn_562727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_562729 = invoke(stypy.reporting.localization.Localization(__file__, 367, 34), getitem___562728, int_562726)
        
        
        # Obtaining an instance of the builtin type 'list' (line 367)
        list_562730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 367)
        # Adding element type (line 367)
        # Getting the type of 'kn0' (line 367)
        kn0_562731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 41), 'kn0', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 40), list_562730, kn0_562731)
        # Adding element type (line 367)
        # Getting the type of 'kn1' (line 367)
        kn1_562732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 45), 'kn1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 40), list_562730, kn1_562732)
        # Adding element type (line 367)
        # Getting the type of 'kn2' (line 367)
        kn2_562733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 49), 'kn2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 367, 40), list_562730, kn2_562733)
        
        int_562734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 54), 'int')
        # Processing the call keyword arguments (line 367)
        kwargs_562735 = {}
        # Getting the type of 'assert_array_almost_equal' (line 367)
        assert_array_almost_equal_562725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 367)
        assert_array_almost_equal_call_result_562736 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), assert_array_almost_equal_562725, *[subscript_call_result_562729, list_562730, int_562734], **kwargs_562735)
        
        
        # ################# End of 'test_sph_kn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sph_kn' in the type store
        # Getting the type of 'stypy_return_type' (line 350)
        stypy_return_type_562737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562737)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sph_kn'
        return stypy_return_type_562737


    @norecursion
    def test_sph_yn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sph_yn'
        module_type_store = module_type_store.open_function_context('test_sph_yn', 369, 4, False)
        # Assigning a type to the variable 'self' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_localization', localization)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_function_name', 'TestSphericalOld.test_sph_yn')
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_param_names_list', [])
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSphericalOld.test_sph_yn.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.test_sph_yn', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sph_yn', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sph_yn(...)' code ##################

        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to spherical_yn(...): (line 370)
        # Processing the call arguments (line 370)
        int_562739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 27), 'int')
        float_562740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 30), 'float')
        # Processing the call keyword arguments (line 370)
        kwargs_562741 = {}
        # Getting the type of 'spherical_yn' (line 370)
        spherical_yn_562738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 14), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 370)
        spherical_yn_call_result_562742 = invoke(stypy.reporting.localization.Localization(__file__, 370, 14), spherical_yn_562738, *[int_562739, float_562740], **kwargs_562741)
        
        # Assigning a type to the variable 'sy1' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'sy1', spherical_yn_call_result_562742)
        
        # Assigning a Call to a Name (line 371):
        
        # Assigning a Call to a Name (line 371):
        
        # Call to spherical_yn(...): (line 371)
        # Processing the call arguments (line 371)
        int_562744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 27), 'int')
        float_562745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 30), 'float')
        # Processing the call keyword arguments (line 371)
        kwargs_562746 = {}
        # Getting the type of 'spherical_yn' (line 371)
        spherical_yn_562743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 371)
        spherical_yn_call_result_562747 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), spherical_yn_562743, *[int_562744, float_562745], **kwargs_562746)
        
        # Assigning a type to the variable 'sy2' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'sy2', spherical_yn_call_result_562747)
        
        # Call to assert_almost_equal(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'sy1' (line 372)
        sy1_562749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 28), 'sy1', False)
        float_562750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 32), 'float')
        int_562751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 43), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_562752 = {}
        # Getting the type of 'assert_almost_equal' (line 372)
        assert_almost_equal_562748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 372)
        assert_almost_equal_call_result_562753 = invoke(stypy.reporting.localization.Localization(__file__, 372, 8), assert_almost_equal_562748, *[sy1_562749, float_562750, int_562751], **kwargs_562752)
        
        
        # Call to assert_almost_equal(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'sy2' (line 373)
        sy2_562755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 28), 'sy2', False)
        float_562756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 32), 'float')
        int_562757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 43), 'int')
        # Processing the call keyword arguments (line 373)
        kwargs_562758 = {}
        # Getting the type of 'assert_almost_equal' (line 373)
        assert_almost_equal_562754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 373)
        assert_almost_equal_call_result_562759 = invoke(stypy.reporting.localization.Localization(__file__, 373, 8), assert_almost_equal_562754, *[sy2_562755, float_562756, int_562757], **kwargs_562758)
        
        
        # Assigning a BinOp to a Name (line 374):
        
        # Assigning a BinOp to a Name (line 374):
        
        # Call to spherical_yn(...): (line 374)
        # Processing the call arguments (line 374)
        int_562761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 30), 'int')
        float_562762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 33), 'float')
        # Processing the call keyword arguments (line 374)
        kwargs_562763 = {}
        # Getting the type of 'spherical_yn' (line 374)
        spherical_yn_562760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 17), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 374)
        spherical_yn_call_result_562764 = invoke(stypy.reporting.localization.Localization(__file__, 374, 17), spherical_yn_562760, *[int_562761, float_562762], **kwargs_562763)
        
        int_562765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 40), 'int')
        
        # Call to spherical_yn(...): (line 374)
        # Processing the call arguments (line 374)
        int_562767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 55), 'int')
        float_562768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 58), 'float')
        # Processing the call keyword arguments (line 374)
        kwargs_562769 = {}
        # Getting the type of 'spherical_yn' (line 374)
        spherical_yn_562766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 42), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 374)
        spherical_yn_call_result_562770 = invoke(stypy.reporting.localization.Localization(__file__, 374, 42), spherical_yn_562766, *[int_562767, float_562768], **kwargs_562769)
        
        # Applying the binary operator '*' (line 374)
        result_mul_562771 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 40), '*', int_562765, spherical_yn_call_result_562770)
        
        # Applying the binary operator '-' (line 374)
        result_sub_562772 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 17), '-', spherical_yn_call_result_562764, result_mul_562771)
        
        int_562773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 64), 'int')
        # Applying the binary operator 'div' (line 374)
        result_div_562774 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 16), 'div', result_sub_562772, int_562773)
        
        # Assigning a type to the variable 'sphpy' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'sphpy', result_div_562774)
        
        # Assigning a Call to a Name (line 375):
        
        # Assigning a Call to a Name (line 375):
        
        # Call to spherical_yn(...): (line 375)
        # Processing the call arguments (line 375)
        int_562776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 27), 'int')
        float_562777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 30), 'float')
        # Processing the call keyword arguments (line 375)
        # Getting the type of 'True' (line 375)
        True_562778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 46), 'True', False)
        keyword_562779 = True_562778
        kwargs_562780 = {'derivative': keyword_562779}
        # Getting the type of 'spherical_yn' (line 375)
        spherical_yn_562775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'spherical_yn', False)
        # Calling spherical_yn(args, kwargs) (line 375)
        spherical_yn_call_result_562781 = invoke(stypy.reporting.localization.Localization(__file__, 375, 14), spherical_yn_562775, *[int_562776, float_562777], **kwargs_562780)
        
        # Assigning a type to the variable 'sy3' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'sy3', spherical_yn_call_result_562781)
        
        # Call to assert_almost_equal(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'sy3' (line 376)
        sy3_562783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 28), 'sy3', False)
        # Getting the type of 'sphpy' (line 376)
        sphpy_562784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 32), 'sphpy', False)
        int_562785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 38), 'int')
        # Processing the call keyword arguments (line 376)
        kwargs_562786 = {}
        # Getting the type of 'assert_almost_equal' (line 376)
        assert_almost_equal_562782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 376)
        assert_almost_equal_call_result_562787 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert_almost_equal_562782, *[sy3_562783, sphpy_562784, int_562785], **kwargs_562786)
        
        
        # ################# End of 'test_sph_yn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sph_yn' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_562788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_562788)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sph_yn'
        return stypy_return_type_562788


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 295, 0, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSphericalOld.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestSphericalOld' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'TestSphericalOld', TestSphericalOld)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
