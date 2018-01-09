
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Tests for the Ellipsoidal Harmonic Function,
3: # Distributed under the same license as SciPy itself.
4: #
5: 
6: from __future__ import division, print_function, absolute_import
7: 
8: import numpy as np
9: from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
10:                            assert_)
11: from scipy._lib._numpy_compat import suppress_warnings
12: from scipy.special._testutils import assert_func_equal
13: from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
14: from scipy.integrate import IntegrationWarning
15: from numpy import sqrt, pi
16: 
17: 
18: def test_ellip_potential():
19:     def change_coefficient(lambda1, mu, nu, h2, k2):
20:         x = sqrt(lambda1**2*mu**2*nu**2/(h2*k2))
21:         y = sqrt((lambda1**2 - h2)*(mu**2 - h2)*(h2 - nu**2)/(h2*(k2 - h2)))
22:         z = sqrt((lambda1**2 - k2)*(k2 - mu**2)*(k2 - nu**2)/(k2*(k2 - h2)))
23:         return x, y, z
24: 
25:     def solid_int_ellip(lambda1, mu, nu, n, p, h2, k2):
26:         return (ellip_harm(h2, k2, n, p, lambda1)*ellip_harm(h2, k2, n, p, mu)
27:                * ellip_harm(h2, k2, n, p, nu))
28: 
29:     def solid_int_ellip2(lambda1, mu, nu, n, p, h2, k2):
30:         return (ellip_harm_2(h2, k2, n, p, lambda1)
31:                 * ellip_harm(h2, k2, n, p, mu)*ellip_harm(h2, k2, n, p, nu))
32: 
33:     def summation(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
34:         tol = 1e-8
35:         sum1 = 0
36:         for n in range(20):
37:             xsum = 0
38:             for p in range(1, 2*n+2):
39:                 xsum += (4*pi*(solid_int_ellip(lambda2, mu2, nu2, n, p, h2, k2)
40:                     * solid_int_ellip2(lambda1, mu1, nu1, n, p, h2, k2)) /
41:                     (ellip_normal(h2, k2, n, p)*(2*n + 1)))
42:             if abs(xsum) < 0.1*tol*abs(sum1):
43:                 break
44:             sum1 += xsum
45:         return sum1, xsum
46: 
47:     def potential(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
48:         x1, y1, z1 = change_coefficient(lambda1, mu1, nu1, h2, k2)
49:         x2, y2, z2 = change_coefficient(lambda2, mu2, nu2, h2, k2)
50:         res = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
51:         return 1/res
52: 
53:     pts = [
54:         (120, sqrt(19), 2, 41, sqrt(17), 2, 15, 25),
55:         (120, sqrt(16), 3.2, 21, sqrt(11), 2.9, 11, 20),
56:        ]
57: 
58:     with suppress_warnings() as sup:
59:         sup.filter(IntegrationWarning, "The occurrence of roundoff error")
60:         sup.filter(IntegrationWarning, "The maximum number of subdivisions")
61: 
62:         for p in pts:
63:             err_msg = repr(p)
64:             exact = potential(*p)
65:             result, last_term = summation(*p)
66:             assert_allclose(exact, result, atol=0, rtol=1e-8, err_msg=err_msg)
67:             assert_(abs(result - exact) < 10*abs(last_term), err_msg)
68: 
69: 
70: def test_ellip_norm():
71: 
72:     def G01(h2, k2):
73:         return 4*pi
74: 
75:     def G11(h2, k2):
76:         return 4*pi*h2*k2/3
77: 
78:     def G12(h2, k2):
79:         return 4*pi*h2*(k2 - h2)/3
80: 
81:     def G13(h2, k2):
82:         return 4*pi*k2*(k2 - h2)/3
83: 
84:     def G22(h2, k2):
85:         res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2 +
86:         sqrt(h2**2 + k2**2 - h2*k2)*(-2*(h2**3 + k2**3) + 3*h2*k2*(h2 + k2)))
87:         return 16*pi/405*res
88: 
89:     def G21(h2, k2):
90:         res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2
91:         + sqrt(h2**2 + k2**2 - h2*k2)*(2*(h2**3 + k2**3) - 3*h2*k2*(h2 + k2)))
92:         return 16*pi/405*res
93: 
94:     def G23(h2, k2):
95:         return 4*pi*h2**2*k2*(k2 - h2)/15
96: 
97:     def G24(h2, k2):
98:         return 4*pi*h2*k2**2*(k2 - h2)/15
99: 
100:     def G25(h2, k2):
101:         return 4*pi*h2*k2*(k2 - h2)**2/15
102: 
103:     def G32(h2, k2):
104:         res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
105:         + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(-8*(h2**3 + k2**3) +
106:         11*h2*k2*(h2 + k2)))
107:         return 16*pi/13125*k2*h2*res
108: 
109:     def G31(h2, k2):
110:         res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
111:         + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(8*(h2**3 + k2**3) -
112:         11*h2*k2*(h2 + k2)))
113:         return 16*pi/13125*h2*k2*res
114: 
115:     def G34(h2, k2):
116:         res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
117:         + sqrt(h2**2 + 4*k2**2 - h2*k2)*(-6*h2**3 - 8*k2**3 + 9*h2**2*k2 +
118:                                             13*h2*k2**2))
119:         return 16*pi/13125*h2*(k2 - h2)*res
120: 
121:     def G33(h2, k2):
122:         res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
123:         + sqrt(h2**2 + 4*k2**2 - h2*k2)*(6*h2**3 + 8*k2**3 - 9*h2**2*k2 -
124:         13*h2*k2**2))
125:         return 16*pi/13125*h2*(k2 - h2)*res
126: 
127:     def G36(h2, k2):
128:         res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
129:         + sqrt(4*h2**2 + k2**2 - h2*k2)*(-8*h2**3 - 6*k2**3 + 13*h2**2*k2 +
130:         9*h2*k2**2))
131:         return 16*pi/13125*k2*(k2 - h2)*res
132: 
133:     def G35(h2, k2):
134:         res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
135:         + sqrt(4*h2**2 + k2**2 - h2*k2)*(8*h2**3 + 6*k2**3 - 13*h2**2*k2 -
136:         9*h2*k2**2))
137:         return 16*pi/13125*k2*(k2 - h2)*res
138: 
139:     def G37(h2, k2):
140:         return 4*pi*h2**2*k2**2*(k2 - h2)**2/105
141: 
142:     known_funcs = {(0, 1): G01, (1, 1): G11, (1, 2): G12, (1, 3): G13,
143:                    (2, 1): G21, (2, 2): G22, (2, 3): G23, (2, 4): G24,
144:                    (2, 5): G25, (3, 1): G31, (3, 2): G32, (3, 3): G33,
145:                    (3, 4): G34, (3, 5): G35, (3, 6): G36, (3, 7): G37}
146: 
147:     def _ellip_norm(n, p, h2, k2):
148:         func = known_funcs[n, p]
149:         return func(h2, k2)
150:     _ellip_norm = np.vectorize(_ellip_norm)
151: 
152:     def ellip_normal_known(h2, k2, n, p):
153:         return _ellip_norm(n, p, h2, k2)
154: 
155:     # generate both large and small h2 < k2 pairs
156:     np.random.seed(1234)
157:     h2 = np.random.pareto(0.5, size=1)
158:     k2 = h2 * (1 + np.random.pareto(0.5, size=h2.size))
159: 
160:     points = []
161:     for n in range(4):
162:         for p in range(1, 2*n+2):
163:             points.append((h2, k2, n*np.ones(h2.size), p*np.ones(h2.size)))
164:     points = np.array(points)
165:     with suppress_warnings() as sup:
166:         sup.filter(IntegrationWarning, "The occurrence of roundoff error")
167:         assert_func_equal(ellip_normal, ellip_normal_known, points, rtol=1e-12)
168: 
169: 
170: def test_ellip_harm_2():
171: 
172:     def I1(h2, k2, s):
173:         res = (ellip_harm_2(h2, k2, 1, 1, s)/(3 * ellip_harm(h2, k2, 1, 1, s))
174:         + ellip_harm_2(h2, k2, 1, 2, s)/(3 * ellip_harm(h2, k2, 1, 2, s)) +
175:         ellip_harm_2(h2, k2, 1, 3, s)/(3 * ellip_harm(h2, k2, 1, 3, s)))
176:         return res
177: 
178:     with suppress_warnings() as sup:
179:         sup.filter(IntegrationWarning, "The occurrence of roundoff error")
180:         assert_almost_equal(I1(5, 8, 10), 1/(10*sqrt((100-5)*(100-8))))
181: 
182:         # Values produced by code from arXiv:1204.0267
183:         assert_almost_equal(ellip_harm_2(5, 8, 2, 1, 10), 0.00108056853382)
184:         assert_almost_equal(ellip_harm_2(5, 8, 2, 2, 10), 0.00105820513809)
185:         assert_almost_equal(ellip_harm_2(5, 8, 2, 3, 10), 0.00106058384743)
186:         assert_almost_equal(ellip_harm_2(5, 8, 2, 4, 10), 0.00106774492306)
187:         assert_almost_equal(ellip_harm_2(5, 8, 2, 5, 10), 0.00107976356454)
188: 
189: 
190: def test_ellip_harm():
191: 
192:     def E01(h2, k2, s):
193:         return 1
194: 
195:     def E11(h2, k2, s):
196:         return s
197: 
198:     def E12(h2, k2, s):
199:         return sqrt(abs(s*s - h2))
200: 
201:     def E13(h2, k2, s):
202:         return sqrt(abs(s*s - k2))
203: 
204:     def E21(h2, k2, s):
205:         return s*s - 1/3*((h2 + k2) + sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))
206: 
207:     def E22(h2, k2, s):
208:         return s*s - 1/3*((h2 + k2) - sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))
209: 
210:     def E23(h2, k2, s):
211:         return s * sqrt(abs(s*s - h2))
212: 
213:     def E24(h2, k2, s):
214:         return s * sqrt(abs(s*s - k2))
215: 
216:     def E25(h2, k2, s):
217:         return sqrt(abs((s*s - h2)*(s*s - k2)))
218: 
219:     def E31(h2, k2, s):
220:         return s*s*s - (s/5)*(2*(h2 + k2) + sqrt(4*(h2 + k2)*(h2 + k2) -
221:         15*h2*k2))
222: 
223:     def E32(h2, k2, s):
224:         return s*s*s - (s/5)*(2*(h2 + k2) - sqrt(4*(h2 + k2)*(h2 + k2) -
225:         15*h2*k2))
226: 
227:     def E33(h2, k2, s):
228:         return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) + sqrt(abs((h2 +
229:         2*k2)*(h2 + 2*k2) - 5*h2*k2))))
230: 
231:     def E34(h2, k2, s):
232:         return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) - sqrt(abs((h2 +
233:         2*k2)*(h2 + 2*k2) - 5*h2*k2))))
234: 
235:     def E35(h2, k2, s):
236:         return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) + sqrt(abs((2*h2
237:         + k2)*(2*h2 + k2) - 5*h2*k2))))
238: 
239:     def E36(h2, k2, s):
240:         return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) - sqrt(abs((2*h2
241:         + k2)*(2*h2 + k2) - 5*h2*k2))))
242: 
243:     def E37(h2, k2, s):
244:         return s * sqrt(abs((s*s - h2)*(s*s - k2)))
245: 
246:     assert_equal(ellip_harm(5, 8, 1, 2, 2.5, 1, 1),
247:     ellip_harm(5, 8, 1, 2, 2.5))
248: 
249:     known_funcs = {(0, 1): E01, (1, 1): E11, (1, 2): E12, (1, 3): E13,
250:                    (2, 1): E21, (2, 2): E22, (2, 3): E23, (2, 4): E24,
251:                    (2, 5): E25, (3, 1): E31, (3, 2): E32, (3, 3): E33,
252:                    (3, 4): E34, (3, 5): E35, (3, 6): E36, (3, 7): E37}
253: 
254:     point_ref = []
255: 
256:     def ellip_harm_known(h2, k2, n, p, s):
257:         for i in range(h2.size):
258:             func = known_funcs[(int(n[i]), int(p[i]))]
259:             point_ref.append(func(h2[i], k2[i], s[i]))
260:         return point_ref
261: 
262:     np.random.seed(1234)
263:     h2 = np.random.pareto(0.5, size=30)
264:     k2 = h2*(1 + np.random.pareto(0.5, size=h2.size))
265:     s = np.random.pareto(0.5, size=h2.size)
266:     points = []
267:     for i in range(h2.size):
268:         for n in range(4):
269:             for p in range(1, 2*n+2):
270:                 points.append((h2[i], k2[i], n, p, s[i]))
271:     points = np.array(points)
272:     assert_func_equal(ellip_harm, ellip_harm_known, points, rtol=1e-12)
273: 
274: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537401 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_537401) is not StypyTypeError):

    if (import_537401 != 'pyd_module'):
        __import__(import_537401)
        sys_modules_537402 = sys.modules[import_537401]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_537402.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_537401)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_allclose, assert_' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_537403) is not StypyTypeError):

    if (import_537403 != 'pyd_module'):
        __import__(import_537403)
        sys_modules_537404 = sys.modules[import_537403]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_537404.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_537404, sys_modules_537404.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_allclose, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_allclose', 'assert_'], [assert_equal, assert_almost_equal, assert_allclose, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_537403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537405 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_537405) is not StypyTypeError):

    if (import_537405 != 'pyd_module'):
        __import__(import_537405)
        sys_modules_537406 = sys.modules[import_537405]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_537406.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_537406, sys_modules_537406.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_537405)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.special._testutils import assert_func_equal' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537407 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils')

if (type(import_537407) is not StypyTypeError):

    if (import_537407 != 'pyd_module'):
        __import__(import_537407)
        sys_modules_537408 = sys.modules[import_537407]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', sys_modules_537408.module_type_store, module_type_store, ['assert_func_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_537408, sys_modules_537408.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import assert_func_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', None, module_type_store, ['assert_func_equal'], [assert_func_equal])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.special._testutils', import_537407)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.special import ellip_harm, ellip_harm_2, ellip_normal' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537409 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special')

if (type(import_537409) is not StypyTypeError):

    if (import_537409 != 'pyd_module'):
        __import__(import_537409)
        sys_modules_537410 = sys.modules[import_537409]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', sys_modules_537410.module_type_store, module_type_store, ['ellip_harm', 'ellip_harm_2', 'ellip_normal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_537410, sys_modules_537410.module_type_store, module_type_store)
    else:
        from scipy.special import ellip_harm, ellip_harm_2, ellip_normal

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', None, module_type_store, ['ellip_harm', 'ellip_harm_2', 'ellip_normal'], [ellip_harm, ellip_harm_2, ellip_normal])

else:
    # Assigning a type to the variable 'scipy.special' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', import_537409)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.integrate import IntegrationWarning' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537411 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.integrate')

if (type(import_537411) is not StypyTypeError):

    if (import_537411 != 'pyd_module'):
        __import__(import_537411)
        sys_modules_537412 = sys.modules[import_537411]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.integrate', sys_modules_537412.module_type_store, module_type_store, ['IntegrationWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_537412, sys_modules_537412.module_type_store, module_type_store)
    else:
        from scipy.integrate import IntegrationWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.integrate', None, module_type_store, ['IntegrationWarning'], [IntegrationWarning])

else:
    # Assigning a type to the variable 'scipy.integrate' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.integrate', import_537411)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy import sqrt, pi' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_537413 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy')

if (type(import_537413) is not StypyTypeError):

    if (import_537413 != 'pyd_module'):
        __import__(import_537413)
        sys_modules_537414 = sys.modules[import_537413]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', sys_modules_537414.module_type_store, module_type_store, ['sqrt', 'pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_537414, sys_modules_537414.module_type_store, module_type_store)
    else:
        from numpy import sqrt, pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', None, module_type_store, ['sqrt', 'pi'], [sqrt, pi])

else:
    # Assigning a type to the variable 'numpy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy', import_537413)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_ellip_potential(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ellip_potential'
    module_type_store = module_type_store.open_function_context('test_ellip_potential', 18, 0, False)
    
    # Passed parameters checking function
    test_ellip_potential.stypy_localization = localization
    test_ellip_potential.stypy_type_of_self = None
    test_ellip_potential.stypy_type_store = module_type_store
    test_ellip_potential.stypy_function_name = 'test_ellip_potential'
    test_ellip_potential.stypy_param_names_list = []
    test_ellip_potential.stypy_varargs_param_name = None
    test_ellip_potential.stypy_kwargs_param_name = None
    test_ellip_potential.stypy_call_defaults = defaults
    test_ellip_potential.stypy_call_varargs = varargs
    test_ellip_potential.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ellip_potential', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ellip_potential', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ellip_potential(...)' code ##################


    @norecursion
    def change_coefficient(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_coefficient'
        module_type_store = module_type_store.open_function_context('change_coefficient', 19, 4, False)
        
        # Passed parameters checking function
        change_coefficient.stypy_localization = localization
        change_coefficient.stypy_type_of_self = None
        change_coefficient.stypy_type_store = module_type_store
        change_coefficient.stypy_function_name = 'change_coefficient'
        change_coefficient.stypy_param_names_list = ['lambda1', 'mu', 'nu', 'h2', 'k2']
        change_coefficient.stypy_varargs_param_name = None
        change_coefficient.stypy_kwargs_param_name = None
        change_coefficient.stypy_call_defaults = defaults
        change_coefficient.stypy_call_varargs = varargs
        change_coefficient.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'change_coefficient', ['lambda1', 'mu', 'nu', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_coefficient', localization, ['lambda1', 'mu', 'nu', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_coefficient(...)' code ##################

        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to sqrt(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'lambda1' (line 20)
        lambda1_537416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'lambda1', False)
        int_537417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'int')
        # Applying the binary operator '**' (line 20)
        result_pow_537418 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 17), '**', lambda1_537416, int_537417)
        
        # Getting the type of 'mu' (line 20)
        mu_537419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 28), 'mu', False)
        int_537420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 32), 'int')
        # Applying the binary operator '**' (line 20)
        result_pow_537421 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 28), '**', mu_537419, int_537420)
        
        # Applying the binary operator '*' (line 20)
        result_mul_537422 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 17), '*', result_pow_537418, result_pow_537421)
        
        # Getting the type of 'nu' (line 20)
        nu_537423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'nu', False)
        int_537424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'int')
        # Applying the binary operator '**' (line 20)
        result_pow_537425 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 34), '**', nu_537423, int_537424)
        
        # Applying the binary operator '*' (line 20)
        result_mul_537426 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 33), '*', result_mul_537422, result_pow_537425)
        
        # Getting the type of 'h2' (line 20)
        h2_537427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 41), 'h2', False)
        # Getting the type of 'k2' (line 20)
        k2_537428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 44), 'k2', False)
        # Applying the binary operator '*' (line 20)
        result_mul_537429 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 41), '*', h2_537427, k2_537428)
        
        # Applying the binary operator 'div' (line 20)
        result_div_537430 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 39), 'div', result_mul_537426, result_mul_537429)
        
        # Processing the call keyword arguments (line 20)
        kwargs_537431 = {}
        # Getting the type of 'sqrt' (line 20)
        sqrt_537415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 20)
        sqrt_call_result_537432 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), sqrt_537415, *[result_div_537430], **kwargs_537431)
        
        # Assigning a type to the variable 'x' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'x', sqrt_call_result_537432)
        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to sqrt(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'lambda1' (line 21)
        lambda1_537434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'lambda1', False)
        int_537435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
        # Applying the binary operator '**' (line 21)
        result_pow_537436 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 18), '**', lambda1_537434, int_537435)
        
        # Getting the type of 'h2' (line 21)
        h2_537437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'h2', False)
        # Applying the binary operator '-' (line 21)
        result_sub_537438 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 18), '-', result_pow_537436, h2_537437)
        
        # Getting the type of 'mu' (line 21)
        mu_537439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'mu', False)
        int_537440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'int')
        # Applying the binary operator '**' (line 21)
        result_pow_537441 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 36), '**', mu_537439, int_537440)
        
        # Getting the type of 'h2' (line 21)
        h2_537442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 44), 'h2', False)
        # Applying the binary operator '-' (line 21)
        result_sub_537443 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 36), '-', result_pow_537441, h2_537442)
        
        # Applying the binary operator '*' (line 21)
        result_mul_537444 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 17), '*', result_sub_537438, result_sub_537443)
        
        # Getting the type of 'h2' (line 21)
        h2_537445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 49), 'h2', False)
        # Getting the type of 'nu' (line 21)
        nu_537446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 54), 'nu', False)
        int_537447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 58), 'int')
        # Applying the binary operator '**' (line 21)
        result_pow_537448 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 54), '**', nu_537446, int_537447)
        
        # Applying the binary operator '-' (line 21)
        result_sub_537449 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 49), '-', h2_537445, result_pow_537448)
        
        # Applying the binary operator '*' (line 21)
        result_mul_537450 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 47), '*', result_mul_537444, result_sub_537449)
        
        # Getting the type of 'h2' (line 21)
        h2_537451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 62), 'h2', False)
        # Getting the type of 'k2' (line 21)
        k2_537452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 66), 'k2', False)
        # Getting the type of 'h2' (line 21)
        h2_537453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 71), 'h2', False)
        # Applying the binary operator '-' (line 21)
        result_sub_537454 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 66), '-', k2_537452, h2_537453)
        
        # Applying the binary operator '*' (line 21)
        result_mul_537455 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 62), '*', h2_537451, result_sub_537454)
        
        # Applying the binary operator 'div' (line 21)
        result_div_537456 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 60), 'div', result_mul_537450, result_mul_537455)
        
        # Processing the call keyword arguments (line 21)
        kwargs_537457 = {}
        # Getting the type of 'sqrt' (line 21)
        sqrt_537433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 21)
        sqrt_call_result_537458 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), sqrt_537433, *[result_div_537456], **kwargs_537457)
        
        # Assigning a type to the variable 'y' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'y', sqrt_call_result_537458)
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to sqrt(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'lambda1' (line 22)
        lambda1_537460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'lambda1', False)
        int_537461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_537462 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 18), '**', lambda1_537460, int_537461)
        
        # Getting the type of 'k2' (line 22)
        k2_537463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'k2', False)
        # Applying the binary operator '-' (line 22)
        result_sub_537464 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 18), '-', result_pow_537462, k2_537463)
        
        # Getting the type of 'k2' (line 22)
        k2_537465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 36), 'k2', False)
        # Getting the type of 'mu' (line 22)
        mu_537466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 41), 'mu', False)
        int_537467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 45), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_537468 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 41), '**', mu_537466, int_537467)
        
        # Applying the binary operator '-' (line 22)
        result_sub_537469 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 36), '-', k2_537465, result_pow_537468)
        
        # Applying the binary operator '*' (line 22)
        result_mul_537470 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 17), '*', result_sub_537464, result_sub_537469)
        
        # Getting the type of 'k2' (line 22)
        k2_537471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'k2', False)
        # Getting the type of 'nu' (line 22)
        nu_537472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 54), 'nu', False)
        int_537473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 58), 'int')
        # Applying the binary operator '**' (line 22)
        result_pow_537474 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 54), '**', nu_537472, int_537473)
        
        # Applying the binary operator '-' (line 22)
        result_sub_537475 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 49), '-', k2_537471, result_pow_537474)
        
        # Applying the binary operator '*' (line 22)
        result_mul_537476 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 47), '*', result_mul_537470, result_sub_537475)
        
        # Getting the type of 'k2' (line 22)
        k2_537477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 62), 'k2', False)
        # Getting the type of 'k2' (line 22)
        k2_537478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 66), 'k2', False)
        # Getting the type of 'h2' (line 22)
        h2_537479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 71), 'h2', False)
        # Applying the binary operator '-' (line 22)
        result_sub_537480 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 66), '-', k2_537478, h2_537479)
        
        # Applying the binary operator '*' (line 22)
        result_mul_537481 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 62), '*', k2_537477, result_sub_537480)
        
        # Applying the binary operator 'div' (line 22)
        result_div_537482 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 60), 'div', result_mul_537476, result_mul_537481)
        
        # Processing the call keyword arguments (line 22)
        kwargs_537483 = {}
        # Getting the type of 'sqrt' (line 22)
        sqrt_537459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 22)
        sqrt_call_result_537484 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), sqrt_537459, *[result_div_537482], **kwargs_537483)
        
        # Assigning a type to the variable 'z' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'z', sqrt_call_result_537484)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_537485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'x' (line 23)
        x_537486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), tuple_537485, x_537486)
        # Adding element type (line 23)
        # Getting the type of 'y' (line 23)
        y_537487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), tuple_537485, y_537487)
        # Adding element type (line 23)
        # Getting the type of 'z' (line 23)
        z_537488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 15), tuple_537485, z_537488)
        
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', tuple_537485)
        
        # ################# End of 'change_coefficient(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_coefficient' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_537489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_coefficient'
        return stypy_return_type_537489

    # Assigning a type to the variable 'change_coefficient' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'change_coefficient', change_coefficient)

    @norecursion
    def solid_int_ellip(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solid_int_ellip'
        module_type_store = module_type_store.open_function_context('solid_int_ellip', 25, 4, False)
        
        # Passed parameters checking function
        solid_int_ellip.stypy_localization = localization
        solid_int_ellip.stypy_type_of_self = None
        solid_int_ellip.stypy_type_store = module_type_store
        solid_int_ellip.stypy_function_name = 'solid_int_ellip'
        solid_int_ellip.stypy_param_names_list = ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2']
        solid_int_ellip.stypy_varargs_param_name = None
        solid_int_ellip.stypy_kwargs_param_name = None
        solid_int_ellip.stypy_call_defaults = defaults
        solid_int_ellip.stypy_call_varargs = varargs
        solid_int_ellip.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'solid_int_ellip', ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solid_int_ellip', localization, ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solid_int_ellip(...)' code ##################

        
        # Call to ellip_harm(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'h2' (line 26)
        h2_537491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'h2', False)
        # Getting the type of 'k2' (line 26)
        k2_537492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'k2', False)
        # Getting the type of 'n' (line 26)
        n_537493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'n', False)
        # Getting the type of 'p' (line 26)
        p_537494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 38), 'p', False)
        # Getting the type of 'lambda1' (line 26)
        lambda1_537495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 41), 'lambda1', False)
        # Processing the call keyword arguments (line 26)
        kwargs_537496 = {}
        # Getting the type of 'ellip_harm' (line 26)
        ellip_harm_537490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 26)
        ellip_harm_call_result_537497 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), ellip_harm_537490, *[h2_537491, k2_537492, n_537493, p_537494, lambda1_537495], **kwargs_537496)
        
        
        # Call to ellip_harm(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'h2' (line 26)
        h2_537499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 61), 'h2', False)
        # Getting the type of 'k2' (line 26)
        k2_537500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 65), 'k2', False)
        # Getting the type of 'n' (line 26)
        n_537501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 69), 'n', False)
        # Getting the type of 'p' (line 26)
        p_537502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 72), 'p', False)
        # Getting the type of 'mu' (line 26)
        mu_537503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 75), 'mu', False)
        # Processing the call keyword arguments (line 26)
        kwargs_537504 = {}
        # Getting the type of 'ellip_harm' (line 26)
        ellip_harm_537498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 50), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 26)
        ellip_harm_call_result_537505 = invoke(stypy.reporting.localization.Localization(__file__, 26, 50), ellip_harm_537498, *[h2_537499, k2_537500, n_537501, p_537502, mu_537503], **kwargs_537504)
        
        # Applying the binary operator '*' (line 26)
        result_mul_537506 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 16), '*', ellip_harm_call_result_537497, ellip_harm_call_result_537505)
        
        
        # Call to ellip_harm(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'h2' (line 27)
        h2_537508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'h2', False)
        # Getting the type of 'k2' (line 27)
        k2_537509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'k2', False)
        # Getting the type of 'n' (line 27)
        n_537510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 36), 'n', False)
        # Getting the type of 'p' (line 27)
        p_537511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 39), 'p', False)
        # Getting the type of 'nu' (line 27)
        nu_537512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 42), 'nu', False)
        # Processing the call keyword arguments (line 27)
        kwargs_537513 = {}
        # Getting the type of 'ellip_harm' (line 27)
        ellip_harm_537507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 27)
        ellip_harm_call_result_537514 = invoke(stypy.reporting.localization.Localization(__file__, 27, 17), ellip_harm_537507, *[h2_537508, k2_537509, n_537510, p_537511, nu_537512], **kwargs_537513)
        
        # Applying the binary operator '*' (line 27)
        result_mul_537515 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), '*', result_mul_537506, ellip_harm_call_result_537514)
        
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', result_mul_537515)
        
        # ################# End of 'solid_int_ellip(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solid_int_ellip' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_537516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537516)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solid_int_ellip'
        return stypy_return_type_537516

    # Assigning a type to the variable 'solid_int_ellip' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'solid_int_ellip', solid_int_ellip)

    @norecursion
    def solid_int_ellip2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'solid_int_ellip2'
        module_type_store = module_type_store.open_function_context('solid_int_ellip2', 29, 4, False)
        
        # Passed parameters checking function
        solid_int_ellip2.stypy_localization = localization
        solid_int_ellip2.stypy_type_of_self = None
        solid_int_ellip2.stypy_type_store = module_type_store
        solid_int_ellip2.stypy_function_name = 'solid_int_ellip2'
        solid_int_ellip2.stypy_param_names_list = ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2']
        solid_int_ellip2.stypy_varargs_param_name = None
        solid_int_ellip2.stypy_kwargs_param_name = None
        solid_int_ellip2.stypy_call_defaults = defaults
        solid_int_ellip2.stypy_call_varargs = varargs
        solid_int_ellip2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'solid_int_ellip2', ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'solid_int_ellip2', localization, ['lambda1', 'mu', 'nu', 'n', 'p', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'solid_int_ellip2(...)' code ##################

        
        # Call to ellip_harm_2(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'h2' (line 30)
        h2_537518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'h2', False)
        # Getting the type of 'k2' (line 30)
        k2_537519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'k2', False)
        # Getting the type of 'n' (line 30)
        n_537520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'n', False)
        # Getting the type of 'p' (line 30)
        p_537521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'p', False)
        # Getting the type of 'lambda1' (line 30)
        lambda1_537522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'lambda1', False)
        # Processing the call keyword arguments (line 30)
        kwargs_537523 = {}
        # Getting the type of 'ellip_harm_2' (line 30)
        ellip_harm_2_537517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 30)
        ellip_harm_2_call_result_537524 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), ellip_harm_2_537517, *[h2_537518, k2_537519, n_537520, p_537521, lambda1_537522], **kwargs_537523)
        
        
        # Call to ellip_harm(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'h2' (line 31)
        h2_537526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'h2', False)
        # Getting the type of 'k2' (line 31)
        k2_537527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'k2', False)
        # Getting the type of 'n' (line 31)
        n_537528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'n', False)
        # Getting the type of 'p' (line 31)
        p_537529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'p', False)
        # Getting the type of 'mu' (line 31)
        mu_537530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'mu', False)
        # Processing the call keyword arguments (line 31)
        kwargs_537531 = {}
        # Getting the type of 'ellip_harm' (line 31)
        ellip_harm_537525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 31)
        ellip_harm_call_result_537532 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), ellip_harm_537525, *[h2_537526, k2_537527, n_537528, p_537529, mu_537530], **kwargs_537531)
        
        # Applying the binary operator '*' (line 30)
        result_mul_537533 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 16), '*', ellip_harm_2_call_result_537524, ellip_harm_call_result_537532)
        
        
        # Call to ellip_harm(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'h2' (line 31)
        h2_537535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 58), 'h2', False)
        # Getting the type of 'k2' (line 31)
        k2_537536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 62), 'k2', False)
        # Getting the type of 'n' (line 31)
        n_537537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 66), 'n', False)
        # Getting the type of 'p' (line 31)
        p_537538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 69), 'p', False)
        # Getting the type of 'nu' (line 31)
        nu_537539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 72), 'nu', False)
        # Processing the call keyword arguments (line 31)
        kwargs_537540 = {}
        # Getting the type of 'ellip_harm' (line 31)
        ellip_harm_537534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 47), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 31)
        ellip_harm_call_result_537541 = invoke(stypy.reporting.localization.Localization(__file__, 31, 47), ellip_harm_537534, *[h2_537535, k2_537536, n_537537, p_537538, nu_537539], **kwargs_537540)
        
        # Applying the binary operator '*' (line 31)
        result_mul_537542 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 46), '*', result_mul_537533, ellip_harm_call_result_537541)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', result_mul_537542)
        
        # ################# End of 'solid_int_ellip2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'solid_int_ellip2' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_537543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537543)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'solid_int_ellip2'
        return stypy_return_type_537543

    # Assigning a type to the variable 'solid_int_ellip2' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'solid_int_ellip2', solid_int_ellip2)

    @norecursion
    def summation(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'summation'
        module_type_store = module_type_store.open_function_context('summation', 33, 4, False)
        
        # Passed parameters checking function
        summation.stypy_localization = localization
        summation.stypy_type_of_self = None
        summation.stypy_type_store = module_type_store
        summation.stypy_function_name = 'summation'
        summation.stypy_param_names_list = ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2']
        summation.stypy_varargs_param_name = None
        summation.stypy_kwargs_param_name = None
        summation.stypy_call_defaults = defaults
        summation.stypy_call_varargs = varargs
        summation.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'summation', ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'summation', localization, ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'summation(...)' code ##################

        
        # Assigning a Num to a Name (line 34):
        
        # Assigning a Num to a Name (line 34):
        float_537544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'float')
        # Assigning a type to the variable 'tol' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tol', float_537544)
        
        # Assigning a Num to a Name (line 35):
        
        # Assigning a Num to a Name (line 35):
        int_537545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'int')
        # Assigning a type to the variable 'sum1' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'sum1', int_537545)
        
        
        # Call to range(...): (line 36)
        # Processing the call arguments (line 36)
        int_537547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'int')
        # Processing the call keyword arguments (line 36)
        kwargs_537548 = {}
        # Getting the type of 'range' (line 36)
        range_537546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'range', False)
        # Calling range(args, kwargs) (line 36)
        range_call_result_537549 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), range_537546, *[int_537547], **kwargs_537548)
        
        # Testing the type of a for loop iterable (line 36)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 8), range_call_result_537549)
        # Getting the type of the for loop variable (line 36)
        for_loop_var_537550 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 8), range_call_result_537549)
        # Assigning a type to the variable 'n' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'n', for_loop_var_537550)
        # SSA begins for a for statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 37):
        
        # Assigning a Num to a Name (line 37):
        int_537551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'int')
        # Assigning a type to the variable 'xsum' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'xsum', int_537551)
        
        
        # Call to range(...): (line 38)
        # Processing the call arguments (line 38)
        int_537553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 27), 'int')
        int_537554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'int')
        # Getting the type of 'n' (line 38)
        n_537555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 32), 'n', False)
        # Applying the binary operator '*' (line 38)
        result_mul_537556 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 30), '*', int_537554, n_537555)
        
        int_537557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
        # Applying the binary operator '+' (line 38)
        result_add_537558 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 30), '+', result_mul_537556, int_537557)
        
        # Processing the call keyword arguments (line 38)
        kwargs_537559 = {}
        # Getting the type of 'range' (line 38)
        range_537552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'range', False)
        # Calling range(args, kwargs) (line 38)
        range_call_result_537560 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), range_537552, *[int_537553, result_add_537558], **kwargs_537559)
        
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_537560)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_537561 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_537560)
        # Assigning a type to the variable 'p' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'p', for_loop_var_537561)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'xsum' (line 39)
        xsum_537562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'xsum')
        int_537563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
        # Getting the type of 'pi' (line 39)
        pi_537564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'pi')
        # Applying the binary operator '*' (line 39)
        result_mul_537565 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 25), '*', int_537563, pi_537564)
        
        
        # Call to solid_int_ellip(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'lambda2' (line 39)
        lambda2_537567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 47), 'lambda2', False)
        # Getting the type of 'mu2' (line 39)
        mu2_537568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'mu2', False)
        # Getting the type of 'nu2' (line 39)
        nu2_537569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 61), 'nu2', False)
        # Getting the type of 'n' (line 39)
        n_537570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 66), 'n', False)
        # Getting the type of 'p' (line 39)
        p_537571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 69), 'p', False)
        # Getting the type of 'h2' (line 39)
        h2_537572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 72), 'h2', False)
        # Getting the type of 'k2' (line 39)
        k2_537573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 76), 'k2', False)
        # Processing the call keyword arguments (line 39)
        kwargs_537574 = {}
        # Getting the type of 'solid_int_ellip' (line 39)
        solid_int_ellip_537566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'solid_int_ellip', False)
        # Calling solid_int_ellip(args, kwargs) (line 39)
        solid_int_ellip_call_result_537575 = invoke(stypy.reporting.localization.Localization(__file__, 39, 31), solid_int_ellip_537566, *[lambda2_537567, mu2_537568, nu2_537569, n_537570, p_537571, h2_537572, k2_537573], **kwargs_537574)
        
        
        # Call to solid_int_ellip2(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'lambda1' (line 40)
        lambda1_537577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'lambda1', False)
        # Getting the type of 'mu1' (line 40)
        mu1_537578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 48), 'mu1', False)
        # Getting the type of 'nu1' (line 40)
        nu1_537579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 53), 'nu1', False)
        # Getting the type of 'n' (line 40)
        n_537580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 58), 'n', False)
        # Getting the type of 'p' (line 40)
        p_537581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 61), 'p', False)
        # Getting the type of 'h2' (line 40)
        h2_537582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 64), 'h2', False)
        # Getting the type of 'k2' (line 40)
        k2_537583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 68), 'k2', False)
        # Processing the call keyword arguments (line 40)
        kwargs_537584 = {}
        # Getting the type of 'solid_int_ellip2' (line 40)
        solid_int_ellip2_537576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'solid_int_ellip2', False)
        # Calling solid_int_ellip2(args, kwargs) (line 40)
        solid_int_ellip2_call_result_537585 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), solid_int_ellip2_537576, *[lambda1_537577, mu1_537578, nu1_537579, n_537580, p_537581, h2_537582, k2_537583], **kwargs_537584)
        
        # Applying the binary operator '*' (line 39)
        result_mul_537586 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 31), '*', solid_int_ellip_call_result_537575, solid_int_ellip2_call_result_537585)
        
        # Applying the binary operator '*' (line 39)
        result_mul_537587 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 29), '*', result_mul_537565, result_mul_537586)
        
        
        # Call to ellip_normal(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'h2' (line 41)
        h2_537589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'h2', False)
        # Getting the type of 'k2' (line 41)
        k2_537590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'k2', False)
        # Getting the type of 'n' (line 41)
        n_537591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 42), 'n', False)
        # Getting the type of 'p' (line 41)
        p_537592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'p', False)
        # Processing the call keyword arguments (line 41)
        kwargs_537593 = {}
        # Getting the type of 'ellip_normal' (line 41)
        ellip_normal_537588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'ellip_normal', False)
        # Calling ellip_normal(args, kwargs) (line 41)
        ellip_normal_call_result_537594 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), ellip_normal_537588, *[h2_537589, k2_537590, n_537591, p_537592], **kwargs_537593)
        
        int_537595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 49), 'int')
        # Getting the type of 'n' (line 41)
        n_537596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 51), 'n')
        # Applying the binary operator '*' (line 41)
        result_mul_537597 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 49), '*', int_537595, n_537596)
        
        int_537598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 55), 'int')
        # Applying the binary operator '+' (line 41)
        result_add_537599 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 49), '+', result_mul_537597, int_537598)
        
        # Applying the binary operator '*' (line 41)
        result_mul_537600 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 21), '*', ellip_normal_call_result_537594, result_add_537599)
        
        # Applying the binary operator 'div' (line 40)
        result_div_537601 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 73), 'div', result_mul_537587, result_mul_537600)
        
        # Applying the binary operator '+=' (line 39)
        result_iadd_537602 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 16), '+=', xsum_537562, result_div_537601)
        # Assigning a type to the variable 'xsum' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'xsum', result_iadd_537602)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'xsum' (line 42)
        xsum_537604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'xsum', False)
        # Processing the call keyword arguments (line 42)
        kwargs_537605 = {}
        # Getting the type of 'abs' (line 42)
        abs_537603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 42)
        abs_call_result_537606 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), abs_537603, *[xsum_537604], **kwargs_537605)
        
        float_537607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'float')
        # Getting the type of 'tol' (line 42)
        tol_537608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'tol')
        # Applying the binary operator '*' (line 42)
        result_mul_537609 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 27), '*', float_537607, tol_537608)
        
        
        # Call to abs(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'sum1' (line 42)
        sum1_537611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'sum1', False)
        # Processing the call keyword arguments (line 42)
        kwargs_537612 = {}
        # Getting the type of 'abs' (line 42)
        abs_537610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 35), 'abs', False)
        # Calling abs(args, kwargs) (line 42)
        abs_call_result_537613 = invoke(stypy.reporting.localization.Localization(__file__, 42, 35), abs_537610, *[sum1_537611], **kwargs_537612)
        
        # Applying the binary operator '*' (line 42)
        result_mul_537614 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 34), '*', result_mul_537609, abs_call_result_537613)
        
        # Applying the binary operator '<' (line 42)
        result_lt_537615 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 15), '<', abs_call_result_537606, result_mul_537614)
        
        # Testing the type of an if condition (line 42)
        if_condition_537616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 12), result_lt_537615)
        # Assigning a type to the variable 'if_condition_537616' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'if_condition_537616', if_condition_537616)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'sum1' (line 44)
        sum1_537617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'sum1')
        # Getting the type of 'xsum' (line 44)
        xsum_537618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'xsum')
        # Applying the binary operator '+=' (line 44)
        result_iadd_537619 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 12), '+=', sum1_537617, xsum_537618)
        # Assigning a type to the variable 'sum1' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'sum1', result_iadd_537619)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_537620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        # Getting the type of 'sum1' (line 45)
        sum1_537621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'sum1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 15), tuple_537620, sum1_537621)
        # Adding element type (line 45)
        # Getting the type of 'xsum' (line 45)
        xsum_537622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'xsum')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 15), tuple_537620, xsum_537622)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', tuple_537620)
        
        # ################# End of 'summation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'summation' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_537623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537623)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'summation'
        return stypy_return_type_537623

    # Assigning a type to the variable 'summation' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'summation', summation)

    @norecursion
    def potential(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'potential'
        module_type_store = module_type_store.open_function_context('potential', 47, 4, False)
        
        # Passed parameters checking function
        potential.stypy_localization = localization
        potential.stypy_type_of_self = None
        potential.stypy_type_store = module_type_store
        potential.stypy_function_name = 'potential'
        potential.stypy_param_names_list = ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2']
        potential.stypy_varargs_param_name = None
        potential.stypy_kwargs_param_name = None
        potential.stypy_call_defaults = defaults
        potential.stypy_call_varargs = varargs
        potential.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'potential', ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'potential', localization, ['lambda1', 'mu1', 'nu1', 'lambda2', 'mu2', 'nu2', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'potential(...)' code ##################

        
        # Assigning a Call to a Tuple (line 48):
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_537624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to change_coefficient(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'lambda1' (line 48)
        lambda1_537626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'lambda1', False)
        # Getting the type of 'mu1' (line 48)
        mu1_537627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'mu1', False)
        # Getting the type of 'nu1' (line 48)
        nu1_537628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 54), 'nu1', False)
        # Getting the type of 'h2' (line 48)
        h2_537629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'h2', False)
        # Getting the type of 'k2' (line 48)
        k2_537630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 63), 'k2', False)
        # Processing the call keyword arguments (line 48)
        kwargs_537631 = {}
        # Getting the type of 'change_coefficient' (line 48)
        change_coefficient_537625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 48)
        change_coefficient_call_result_537632 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), change_coefficient_537625, *[lambda1_537626, mu1_537627, nu1_537628, h2_537629, k2_537630], **kwargs_537631)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___537633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), change_coefficient_call_result_537632, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_537634 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___537633, int_537624)
        
        # Assigning a type to the variable 'tuple_var_assignment_537393' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537393', subscript_call_result_537634)
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_537635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to change_coefficient(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'lambda1' (line 48)
        lambda1_537637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'lambda1', False)
        # Getting the type of 'mu1' (line 48)
        mu1_537638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'mu1', False)
        # Getting the type of 'nu1' (line 48)
        nu1_537639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 54), 'nu1', False)
        # Getting the type of 'h2' (line 48)
        h2_537640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'h2', False)
        # Getting the type of 'k2' (line 48)
        k2_537641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 63), 'k2', False)
        # Processing the call keyword arguments (line 48)
        kwargs_537642 = {}
        # Getting the type of 'change_coefficient' (line 48)
        change_coefficient_537636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 48)
        change_coefficient_call_result_537643 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), change_coefficient_537636, *[lambda1_537637, mu1_537638, nu1_537639, h2_537640, k2_537641], **kwargs_537642)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___537644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), change_coefficient_call_result_537643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_537645 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___537644, int_537635)
        
        # Assigning a type to the variable 'tuple_var_assignment_537394' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537394', subscript_call_result_537645)
        
        # Assigning a Subscript to a Name (line 48):
        
        # Obtaining the type of the subscript
        int_537646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'int')
        
        # Call to change_coefficient(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'lambda1' (line 48)
        lambda1_537648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'lambda1', False)
        # Getting the type of 'mu1' (line 48)
        mu1_537649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'mu1', False)
        # Getting the type of 'nu1' (line 48)
        nu1_537650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 54), 'nu1', False)
        # Getting the type of 'h2' (line 48)
        h2_537651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'h2', False)
        # Getting the type of 'k2' (line 48)
        k2_537652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 63), 'k2', False)
        # Processing the call keyword arguments (line 48)
        kwargs_537653 = {}
        # Getting the type of 'change_coefficient' (line 48)
        change_coefficient_537647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 48)
        change_coefficient_call_result_537654 = invoke(stypy.reporting.localization.Localization(__file__, 48, 21), change_coefficient_537647, *[lambda1_537648, mu1_537649, nu1_537650, h2_537651, k2_537652], **kwargs_537653)
        
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___537655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), change_coefficient_call_result_537654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_537656 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), getitem___537655, int_537646)
        
        # Assigning a type to the variable 'tuple_var_assignment_537395' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537395', subscript_call_result_537656)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_537393' (line 48)
        tuple_var_assignment_537393_537657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537393')
        # Assigning a type to the variable 'x1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'x1', tuple_var_assignment_537393_537657)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_537394' (line 48)
        tuple_var_assignment_537394_537658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537394')
        # Assigning a type to the variable 'y1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'y1', tuple_var_assignment_537394_537658)
        
        # Assigning a Name to a Name (line 48):
        # Getting the type of 'tuple_var_assignment_537395' (line 48)
        tuple_var_assignment_537395_537659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tuple_var_assignment_537395')
        # Assigning a type to the variable 'z1' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'z1', tuple_var_assignment_537395_537659)
        
        # Assigning a Call to a Tuple (line 49):
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_537660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to change_coefficient(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'lambda2' (line 49)
        lambda2_537662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'lambda2', False)
        # Getting the type of 'mu2' (line 49)
        mu2_537663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'mu2', False)
        # Getting the type of 'nu2' (line 49)
        nu2_537664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 54), 'nu2', False)
        # Getting the type of 'h2' (line 49)
        h2_537665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 59), 'h2', False)
        # Getting the type of 'k2' (line 49)
        k2_537666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 63), 'k2', False)
        # Processing the call keyword arguments (line 49)
        kwargs_537667 = {}
        # Getting the type of 'change_coefficient' (line 49)
        change_coefficient_537661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 49)
        change_coefficient_call_result_537668 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), change_coefficient_537661, *[lambda2_537662, mu2_537663, nu2_537664, h2_537665, k2_537666], **kwargs_537667)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___537669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), change_coefficient_call_result_537668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_537670 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___537669, int_537660)
        
        # Assigning a type to the variable 'tuple_var_assignment_537396' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537396', subscript_call_result_537670)
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_537671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to change_coefficient(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'lambda2' (line 49)
        lambda2_537673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'lambda2', False)
        # Getting the type of 'mu2' (line 49)
        mu2_537674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'mu2', False)
        # Getting the type of 'nu2' (line 49)
        nu2_537675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 54), 'nu2', False)
        # Getting the type of 'h2' (line 49)
        h2_537676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 59), 'h2', False)
        # Getting the type of 'k2' (line 49)
        k2_537677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 63), 'k2', False)
        # Processing the call keyword arguments (line 49)
        kwargs_537678 = {}
        # Getting the type of 'change_coefficient' (line 49)
        change_coefficient_537672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 49)
        change_coefficient_call_result_537679 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), change_coefficient_537672, *[lambda2_537673, mu2_537674, nu2_537675, h2_537676, k2_537677], **kwargs_537678)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___537680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), change_coefficient_call_result_537679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_537681 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___537680, int_537671)
        
        # Assigning a type to the variable 'tuple_var_assignment_537397' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537397', subscript_call_result_537681)
        
        # Assigning a Subscript to a Name (line 49):
        
        # Obtaining the type of the subscript
        int_537682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'int')
        
        # Call to change_coefficient(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'lambda2' (line 49)
        lambda2_537684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 40), 'lambda2', False)
        # Getting the type of 'mu2' (line 49)
        mu2_537685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'mu2', False)
        # Getting the type of 'nu2' (line 49)
        nu2_537686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 54), 'nu2', False)
        # Getting the type of 'h2' (line 49)
        h2_537687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 59), 'h2', False)
        # Getting the type of 'k2' (line 49)
        k2_537688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 63), 'k2', False)
        # Processing the call keyword arguments (line 49)
        kwargs_537689 = {}
        # Getting the type of 'change_coefficient' (line 49)
        change_coefficient_537683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'change_coefficient', False)
        # Calling change_coefficient(args, kwargs) (line 49)
        change_coefficient_call_result_537690 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), change_coefficient_537683, *[lambda2_537684, mu2_537685, nu2_537686, h2_537687, k2_537688], **kwargs_537689)
        
        # Obtaining the member '__getitem__' of a type (line 49)
        getitem___537691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), change_coefficient_call_result_537690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 49)
        subscript_call_result_537692 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), getitem___537691, int_537682)
        
        # Assigning a type to the variable 'tuple_var_assignment_537398' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537398', subscript_call_result_537692)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_537396' (line 49)
        tuple_var_assignment_537396_537693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537396')
        # Assigning a type to the variable 'x2' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'x2', tuple_var_assignment_537396_537693)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_537397' (line 49)
        tuple_var_assignment_537397_537694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537397')
        # Assigning a type to the variable 'y2' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'y2', tuple_var_assignment_537397_537694)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_var_assignment_537398' (line 49)
        tuple_var_assignment_537398_537695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_var_assignment_537398')
        # Assigning a type to the variable 'z2' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'z2', tuple_var_assignment_537398_537695)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to sqrt(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'x2' (line 50)
        x2_537697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'x2', False)
        # Getting the type of 'x1' (line 50)
        x1_537698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'x1', False)
        # Applying the binary operator '-' (line 50)
        result_sub_537699 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 20), '-', x2_537697, x1_537698)
        
        int_537700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
        # Applying the binary operator '**' (line 50)
        result_pow_537701 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 19), '**', result_sub_537699, int_537700)
        
        # Getting the type of 'y2' (line 50)
        y2_537702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'y2', False)
        # Getting the type of 'y1' (line 50)
        y1_537703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'y1', False)
        # Applying the binary operator '-' (line 50)
        result_sub_537704 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 35), '-', y2_537702, y1_537703)
        
        int_537705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 45), 'int')
        # Applying the binary operator '**' (line 50)
        result_pow_537706 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 34), '**', result_sub_537704, int_537705)
        
        # Applying the binary operator '+' (line 50)
        result_add_537707 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 19), '+', result_pow_537701, result_pow_537706)
        
        # Getting the type of 'z2' (line 50)
        z2_537708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 50), 'z2', False)
        # Getting the type of 'z1' (line 50)
        z1_537709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 55), 'z1', False)
        # Applying the binary operator '-' (line 50)
        result_sub_537710 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 50), '-', z2_537708, z1_537709)
        
        int_537711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 60), 'int')
        # Applying the binary operator '**' (line 50)
        result_pow_537712 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 49), '**', result_sub_537710, int_537711)
        
        # Applying the binary operator '+' (line 50)
        result_add_537713 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 47), '+', result_add_537707, result_pow_537712)
        
        # Processing the call keyword arguments (line 50)
        kwargs_537714 = {}
        # Getting the type of 'sqrt' (line 50)
        sqrt_537696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 50)
        sqrt_call_result_537715 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), sqrt_537696, *[result_add_537713], **kwargs_537714)
        
        # Assigning a type to the variable 'res' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'res', sqrt_call_result_537715)
        int_537716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'res' (line 51)
        res_537717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'res')
        # Applying the binary operator 'div' (line 51)
        result_div_537718 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), 'div', int_537716, res_537717)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', result_div_537718)
        
        # ################# End of 'potential(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'potential' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_537719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'potential'
        return stypy_return_type_537719

    # Assigning a type to the variable 'potential' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'potential', potential)
    
    # Assigning a List to a Name (line 53):
    
    # Assigning a List to a Name (line 53):
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_537720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_537721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    int_537722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537722)
    # Adding element type (line 54)
    
    # Call to sqrt(...): (line 54)
    # Processing the call arguments (line 54)
    int_537724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_537725 = {}
    # Getting the type of 'sqrt' (line 54)
    sqrt_537723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 14), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 54)
    sqrt_call_result_537726 = invoke(stypy.reporting.localization.Localization(__file__, 54, 14), sqrt_537723, *[int_537724], **kwargs_537725)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, sqrt_call_result_537726)
    # Adding element type (line 54)
    int_537727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537727)
    # Adding element type (line 54)
    int_537728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537728)
    # Adding element type (line 54)
    
    # Call to sqrt(...): (line 54)
    # Processing the call arguments (line 54)
    int_537730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 36), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_537731 = {}
    # Getting the type of 'sqrt' (line 54)
    sqrt_537729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 54)
    sqrt_call_result_537732 = invoke(stypy.reporting.localization.Localization(__file__, 54, 31), sqrt_537729, *[int_537730], **kwargs_537731)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, sqrt_call_result_537732)
    # Adding element type (line 54)
    int_537733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537733)
    # Adding element type (line 54)
    int_537734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537734)
    # Adding element type (line 54)
    int_537735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_537721, int_537735)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_537720, tuple_537721)
    # Adding element type (line 53)
    
    # Obtaining an instance of the builtin type 'tuple' (line 55)
    tuple_537736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 55)
    # Adding element type (line 55)
    int_537737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, int_537737)
    # Adding element type (line 55)
    
    # Call to sqrt(...): (line 55)
    # Processing the call arguments (line 55)
    int_537739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_537740 = {}
    # Getting the type of 'sqrt' (line 55)
    sqrt_537738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 55)
    sqrt_call_result_537741 = invoke(stypy.reporting.localization.Localization(__file__, 55, 14), sqrt_537738, *[int_537739], **kwargs_537740)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, sqrt_call_result_537741)
    # Adding element type (line 55)
    float_537742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, float_537742)
    # Adding element type (line 55)
    int_537743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, int_537743)
    # Adding element type (line 55)
    
    # Call to sqrt(...): (line 55)
    # Processing the call arguments (line 55)
    int_537745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 38), 'int')
    # Processing the call keyword arguments (line 55)
    kwargs_537746 = {}
    # Getting the type of 'sqrt' (line 55)
    sqrt_537744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 55)
    sqrt_call_result_537747 = invoke(stypy.reporting.localization.Localization(__file__, 55, 33), sqrt_537744, *[int_537745], **kwargs_537746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, sqrt_call_result_537747)
    # Adding element type (line 55)
    float_537748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, float_537748)
    # Adding element type (line 55)
    int_537749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 48), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, int_537749)
    # Adding element type (line 55)
    int_537750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 9), tuple_537736, int_537750)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 10), list_537720, tuple_537736)
    
    # Assigning a type to the variable 'pts' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'pts', list_537720)
    
    # Call to suppress_warnings(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_537752 = {}
    # Getting the type of 'suppress_warnings' (line 58)
    suppress_warnings_537751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 58)
    suppress_warnings_call_result_537753 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), suppress_warnings_537751, *[], **kwargs_537752)
    
    with_537754 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 58, 9), suppress_warnings_call_result_537753, 'with parameter', '__enter__', '__exit__')

    if with_537754:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 58)
        enter___537755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), suppress_warnings_call_result_537753, '__enter__')
        with_enter_537756 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), enter___537755)
        # Assigning a type to the variable 'sup' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'sup', with_enter_537756)
        
        # Call to filter(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'IntegrationWarning' (line 59)
        IntegrationWarning_537759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 19), 'IntegrationWarning', False)
        str_537760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 39), 'str', 'The occurrence of roundoff error')
        # Processing the call keyword arguments (line 59)
        kwargs_537761 = {}
        # Getting the type of 'sup' (line 59)
        sup_537757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 59)
        filter_537758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), sup_537757, 'filter')
        # Calling filter(args, kwargs) (line 59)
        filter_call_result_537762 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), filter_537758, *[IntegrationWarning_537759, str_537760], **kwargs_537761)
        
        
        # Call to filter(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'IntegrationWarning' (line 60)
        IntegrationWarning_537765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'IntegrationWarning', False)
        str_537766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'str', 'The maximum number of subdivisions')
        # Processing the call keyword arguments (line 60)
        kwargs_537767 = {}
        # Getting the type of 'sup' (line 60)
        sup_537763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 60)
        filter_537764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), sup_537763, 'filter')
        # Calling filter(args, kwargs) (line 60)
        filter_call_result_537768 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), filter_537764, *[IntegrationWarning_537765, str_537766], **kwargs_537767)
        
        
        # Getting the type of 'pts' (line 62)
        pts_537769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'pts')
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), pts_537769)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_537770 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), pts_537769)
        # Assigning a type to the variable 'p' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'p', for_loop_var_537770)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to repr(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'p' (line 63)
        p_537772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 27), 'p', False)
        # Processing the call keyword arguments (line 63)
        kwargs_537773 = {}
        # Getting the type of 'repr' (line 63)
        repr_537771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'repr', False)
        # Calling repr(args, kwargs) (line 63)
        repr_call_result_537774 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), repr_537771, *[p_537772], **kwargs_537773)
        
        # Assigning a type to the variable 'err_msg' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'err_msg', repr_call_result_537774)
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to potential(...): (line 64)
        # Getting the type of 'p' (line 64)
        p_537776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'p', False)
        # Processing the call keyword arguments (line 64)
        kwargs_537777 = {}
        # Getting the type of 'potential' (line 64)
        potential_537775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'potential', False)
        # Calling potential(args, kwargs) (line 64)
        potential_call_result_537778 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), potential_537775, *[p_537776], **kwargs_537777)
        
        # Assigning a type to the variable 'exact' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'exact', potential_call_result_537778)
        
        # Assigning a Call to a Tuple (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_537779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'int')
        
        # Call to summation(...): (line 65)
        # Getting the type of 'p' (line 65)
        p_537781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 43), 'p', False)
        # Processing the call keyword arguments (line 65)
        kwargs_537782 = {}
        # Getting the type of 'summation' (line 65)
        summation_537780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'summation', False)
        # Calling summation(args, kwargs) (line 65)
        summation_call_result_537783 = invoke(stypy.reporting.localization.Localization(__file__, 65, 32), summation_537780, *[p_537781], **kwargs_537782)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___537784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), summation_call_result_537783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_537785 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), getitem___537784, int_537779)
        
        # Assigning a type to the variable 'tuple_var_assignment_537399' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'tuple_var_assignment_537399', subscript_call_result_537785)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_537786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'int')
        
        # Call to summation(...): (line 65)
        # Getting the type of 'p' (line 65)
        p_537788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 43), 'p', False)
        # Processing the call keyword arguments (line 65)
        kwargs_537789 = {}
        # Getting the type of 'summation' (line 65)
        summation_537787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'summation', False)
        # Calling summation(args, kwargs) (line 65)
        summation_call_result_537790 = invoke(stypy.reporting.localization.Localization(__file__, 65, 32), summation_537787, *[p_537788], **kwargs_537789)
        
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___537791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), summation_call_result_537790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_537792 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), getitem___537791, int_537786)
        
        # Assigning a type to the variable 'tuple_var_assignment_537400' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'tuple_var_assignment_537400', subscript_call_result_537792)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'tuple_var_assignment_537399' (line 65)
        tuple_var_assignment_537399_537793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'tuple_var_assignment_537399')
        # Assigning a type to the variable 'result' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'result', tuple_var_assignment_537399_537793)
        
        # Assigning a Name to a Name (line 65):
        # Getting the type of 'tuple_var_assignment_537400' (line 65)
        tuple_var_assignment_537400_537794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'tuple_var_assignment_537400')
        # Assigning a type to the variable 'last_term' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'last_term', tuple_var_assignment_537400_537794)
        
        # Call to assert_allclose(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'exact' (line 66)
        exact_537796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'exact', False)
        # Getting the type of 'result' (line 66)
        result_537797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'result', False)
        # Processing the call keyword arguments (line 66)
        int_537798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 48), 'int')
        keyword_537799 = int_537798
        float_537800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 56), 'float')
        keyword_537801 = float_537800
        # Getting the type of 'err_msg' (line 66)
        err_msg_537802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 70), 'err_msg', False)
        keyword_537803 = err_msg_537802
        kwargs_537804 = {'rtol': keyword_537801, 'err_msg': keyword_537803, 'atol': keyword_537799}
        # Getting the type of 'assert_allclose' (line 66)
        assert_allclose_537795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 66)
        assert_allclose_call_result_537805 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), assert_allclose_537795, *[exact_537796, result_537797], **kwargs_537804)
        
        
        # Call to assert_(...): (line 67)
        # Processing the call arguments (line 67)
        
        
        # Call to abs(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'result' (line 67)
        result_537808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'result', False)
        # Getting the type of 'exact' (line 67)
        exact_537809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'exact', False)
        # Applying the binary operator '-' (line 67)
        result_sub_537810 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 24), '-', result_537808, exact_537809)
        
        # Processing the call keyword arguments (line 67)
        kwargs_537811 = {}
        # Getting the type of 'abs' (line 67)
        abs_537807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 67)
        abs_call_result_537812 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), abs_537807, *[result_sub_537810], **kwargs_537811)
        
        int_537813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 42), 'int')
        
        # Call to abs(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'last_term' (line 67)
        last_term_537815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 49), 'last_term', False)
        # Processing the call keyword arguments (line 67)
        kwargs_537816 = {}
        # Getting the type of 'abs' (line 67)
        abs_537814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'abs', False)
        # Calling abs(args, kwargs) (line 67)
        abs_call_result_537817 = invoke(stypy.reporting.localization.Localization(__file__, 67, 45), abs_537814, *[last_term_537815], **kwargs_537816)
        
        # Applying the binary operator '*' (line 67)
        result_mul_537818 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 42), '*', int_537813, abs_call_result_537817)
        
        # Applying the binary operator '<' (line 67)
        result_lt_537819 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 20), '<', abs_call_result_537812, result_mul_537818)
        
        # Getting the type of 'err_msg' (line 67)
        err_msg_537820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 61), 'err_msg', False)
        # Processing the call keyword arguments (line 67)
        kwargs_537821 = {}
        # Getting the type of 'assert_' (line 67)
        assert__537806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 67)
        assert__call_result_537822 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), assert__537806, *[result_lt_537819, err_msg_537820], **kwargs_537821)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 58)
        exit___537823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), suppress_warnings_call_result_537753, '__exit__')
        with_exit_537824 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), exit___537823, None, None, None)

    
    # ################# End of 'test_ellip_potential(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ellip_potential' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_537825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_537825)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ellip_potential'
    return stypy_return_type_537825

# Assigning a type to the variable 'test_ellip_potential' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_ellip_potential', test_ellip_potential)

@norecursion
def test_ellip_norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ellip_norm'
    module_type_store = module_type_store.open_function_context('test_ellip_norm', 70, 0, False)
    
    # Passed parameters checking function
    test_ellip_norm.stypy_localization = localization
    test_ellip_norm.stypy_type_of_self = None
    test_ellip_norm.stypy_type_store = module_type_store
    test_ellip_norm.stypy_function_name = 'test_ellip_norm'
    test_ellip_norm.stypy_param_names_list = []
    test_ellip_norm.stypy_varargs_param_name = None
    test_ellip_norm.stypy_kwargs_param_name = None
    test_ellip_norm.stypy_call_defaults = defaults
    test_ellip_norm.stypy_call_varargs = varargs
    test_ellip_norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ellip_norm', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ellip_norm', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ellip_norm(...)' code ##################


    @norecursion
    def G01(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G01'
        module_type_store = module_type_store.open_function_context('G01', 72, 4, False)
        
        # Passed parameters checking function
        G01.stypy_localization = localization
        G01.stypy_type_of_self = None
        G01.stypy_type_store = module_type_store
        G01.stypy_function_name = 'G01'
        G01.stypy_param_names_list = ['h2', 'k2']
        G01.stypy_varargs_param_name = None
        G01.stypy_kwargs_param_name = None
        G01.stypy_call_defaults = defaults
        G01.stypy_call_varargs = varargs
        G01.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G01', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G01', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G01(...)' code ##################

        int_537826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 15), 'int')
        # Getting the type of 'pi' (line 73)
        pi_537827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'pi')
        # Applying the binary operator '*' (line 73)
        result_mul_537828 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 15), '*', int_537826, pi_537827)
        
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', result_mul_537828)
        
        # ################# End of 'G01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G01' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_537829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537829)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G01'
        return stypy_return_type_537829

    # Assigning a type to the variable 'G01' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'G01', G01)

    @norecursion
    def G11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G11'
        module_type_store = module_type_store.open_function_context('G11', 75, 4, False)
        
        # Passed parameters checking function
        G11.stypy_localization = localization
        G11.stypy_type_of_self = None
        G11.stypy_type_store = module_type_store
        G11.stypy_function_name = 'G11'
        G11.stypy_param_names_list = ['h2', 'k2']
        G11.stypy_varargs_param_name = None
        G11.stypy_kwargs_param_name = None
        G11.stypy_call_defaults = defaults
        G11.stypy_call_varargs = varargs
        G11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G11', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G11', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G11(...)' code ##################

        int_537830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'int')
        # Getting the type of 'pi' (line 76)
        pi_537831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'pi')
        # Applying the binary operator '*' (line 76)
        result_mul_537832 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '*', int_537830, pi_537831)
        
        # Getting the type of 'h2' (line 76)
        h2_537833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'h2')
        # Applying the binary operator '*' (line 76)
        result_mul_537834 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 19), '*', result_mul_537832, h2_537833)
        
        # Getting the type of 'k2' (line 76)
        k2_537835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'k2')
        # Applying the binary operator '*' (line 76)
        result_mul_537836 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 22), '*', result_mul_537834, k2_537835)
        
        int_537837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'int')
        # Applying the binary operator 'div' (line 76)
        result_div_537838 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 25), 'div', result_mul_537836, int_537837)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', result_div_537838)
        
        # ################# End of 'G11(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G11' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_537839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537839)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G11'
        return stypy_return_type_537839

    # Assigning a type to the variable 'G11' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'G11', G11)

    @norecursion
    def G12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G12'
        module_type_store = module_type_store.open_function_context('G12', 78, 4, False)
        
        # Passed parameters checking function
        G12.stypy_localization = localization
        G12.stypy_type_of_self = None
        G12.stypy_type_store = module_type_store
        G12.stypy_function_name = 'G12'
        G12.stypy_param_names_list = ['h2', 'k2']
        G12.stypy_varargs_param_name = None
        G12.stypy_kwargs_param_name = None
        G12.stypy_call_defaults = defaults
        G12.stypy_call_varargs = varargs
        G12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G12', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G12', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G12(...)' code ##################

        int_537840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 15), 'int')
        # Getting the type of 'pi' (line 79)
        pi_537841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'pi')
        # Applying the binary operator '*' (line 79)
        result_mul_537842 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '*', int_537840, pi_537841)
        
        # Getting the type of 'h2' (line 79)
        h2_537843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'h2')
        # Applying the binary operator '*' (line 79)
        result_mul_537844 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 19), '*', result_mul_537842, h2_537843)
        
        # Getting the type of 'k2' (line 79)
        k2_537845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'k2')
        # Getting the type of 'h2' (line 79)
        h2_537846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'h2')
        # Applying the binary operator '-' (line 79)
        result_sub_537847 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 24), '-', k2_537845, h2_537846)
        
        # Applying the binary operator '*' (line 79)
        result_mul_537848 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 22), '*', result_mul_537844, result_sub_537847)
        
        int_537849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 33), 'int')
        # Applying the binary operator 'div' (line 79)
        result_div_537850 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 32), 'div', result_mul_537848, int_537849)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', result_div_537850)
        
        # ################# End of 'G12(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G12' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_537851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G12'
        return stypy_return_type_537851

    # Assigning a type to the variable 'G12' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'G12', G12)

    @norecursion
    def G13(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G13'
        module_type_store = module_type_store.open_function_context('G13', 81, 4, False)
        
        # Passed parameters checking function
        G13.stypy_localization = localization
        G13.stypy_type_of_self = None
        G13.stypy_type_store = module_type_store
        G13.stypy_function_name = 'G13'
        G13.stypy_param_names_list = ['h2', 'k2']
        G13.stypy_varargs_param_name = None
        G13.stypy_kwargs_param_name = None
        G13.stypy_call_defaults = defaults
        G13.stypy_call_varargs = varargs
        G13.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G13', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G13', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G13(...)' code ##################

        int_537852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'int')
        # Getting the type of 'pi' (line 82)
        pi_537853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'pi')
        # Applying the binary operator '*' (line 82)
        result_mul_537854 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), '*', int_537852, pi_537853)
        
        # Getting the type of 'k2' (line 82)
        k2_537855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'k2')
        # Applying the binary operator '*' (line 82)
        result_mul_537856 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '*', result_mul_537854, k2_537855)
        
        # Getting the type of 'k2' (line 82)
        k2_537857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'k2')
        # Getting the type of 'h2' (line 82)
        h2_537858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'h2')
        # Applying the binary operator '-' (line 82)
        result_sub_537859 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 24), '-', k2_537857, h2_537858)
        
        # Applying the binary operator '*' (line 82)
        result_mul_537860 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 22), '*', result_mul_537856, result_sub_537859)
        
        int_537861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'int')
        # Applying the binary operator 'div' (line 82)
        result_div_537862 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 32), 'div', result_mul_537860, int_537861)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', result_div_537862)
        
        # ################# End of 'G13(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G13' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_537863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537863)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G13'
        return stypy_return_type_537863

    # Assigning a type to the variable 'G13' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'G13', G13)

    @norecursion
    def G22(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G22'
        module_type_store = module_type_store.open_function_context('G22', 84, 4, False)
        
        # Passed parameters checking function
        G22.stypy_localization = localization
        G22.stypy_type_of_self = None
        G22.stypy_type_store = module_type_store
        G22.stypy_function_name = 'G22'
        G22.stypy_param_names_list = ['h2', 'k2']
        G22.stypy_varargs_param_name = None
        G22.stypy_kwargs_param_name = None
        G22.stypy_call_defaults = defaults
        G22.stypy_call_varargs = varargs
        G22.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G22', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G22', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G22(...)' code ##################

        
        # Assigning a BinOp to a Name (line 85):
        
        # Assigning a BinOp to a Name (line 85):
        int_537864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 15), 'int')
        # Getting the type of 'h2' (line 85)
        h2_537865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'h2')
        int_537866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537867 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 18), '**', h2_537865, int_537866)
        
        # Getting the type of 'k2' (line 85)
        k2_537868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'k2')
        int_537869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537870 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 26), '**', k2_537868, int_537869)
        
        # Applying the binary operator '+' (line 85)
        result_add_537871 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 18), '+', result_pow_537867, result_pow_537870)
        
        # Applying the binary operator '*' (line 85)
        result_mul_537872 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '*', int_537864, result_add_537871)
        
        int_537873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
        # Getting the type of 'h2' (line 85)
        h2_537874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 37), 'h2')
        # Applying the binary operator '*' (line 85)
        result_mul_537875 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 35), '*', int_537873, h2_537874)
        
        # Getting the type of 'k2' (line 85)
        k2_537876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 40), 'k2')
        # Applying the binary operator '*' (line 85)
        result_mul_537877 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 39), '*', result_mul_537875, k2_537876)
        
        # Getting the type of 'h2' (line 85)
        h2_537878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 44), 'h2')
        int_537879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 48), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537880 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 44), '**', h2_537878, int_537879)
        
        # Getting the type of 'k2' (line 85)
        k2_537881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 52), 'k2')
        int_537882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 56), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537883 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 52), '**', k2_537881, int_537882)
        
        # Applying the binary operator '+' (line 85)
        result_add_537884 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 44), '+', result_pow_537880, result_pow_537883)
        
        # Applying the binary operator '*' (line 85)
        result_mul_537885 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 42), '*', result_mul_537877, result_add_537884)
        
        # Applying the binary operator '-' (line 85)
        result_sub_537886 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '-', result_mul_537872, result_mul_537885)
        
        int_537887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 61), 'int')
        # Getting the type of 'h2' (line 85)
        h2_537888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 63), 'h2')
        int_537889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 67), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537890 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 63), '**', h2_537888, int_537889)
        
        # Applying the binary operator '*' (line 85)
        result_mul_537891 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 61), '*', int_537887, result_pow_537890)
        
        # Getting the type of 'k2' (line 85)
        k2_537892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 69), 'k2')
        int_537893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 73), 'int')
        # Applying the binary operator '**' (line 85)
        result_pow_537894 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 69), '**', k2_537892, int_537893)
        
        # Applying the binary operator '*' (line 85)
        result_mul_537895 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 68), '*', result_mul_537891, result_pow_537894)
        
        # Applying the binary operator '+' (line 85)
        result_add_537896 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 59), '+', result_sub_537886, result_mul_537895)
        
        
        # Call to sqrt(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'h2' (line 86)
        h2_537898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'h2', False)
        int_537899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 17), 'int')
        # Applying the binary operator '**' (line 86)
        result_pow_537900 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 13), '**', h2_537898, int_537899)
        
        # Getting the type of 'k2' (line 86)
        k2_537901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'k2', False)
        int_537902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'int')
        # Applying the binary operator '**' (line 86)
        result_pow_537903 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '**', k2_537901, int_537902)
        
        # Applying the binary operator '+' (line 86)
        result_add_537904 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 13), '+', result_pow_537900, result_pow_537903)
        
        # Getting the type of 'h2' (line 86)
        h2_537905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'h2', False)
        # Getting the type of 'k2' (line 86)
        k2_537906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'k2', False)
        # Applying the binary operator '*' (line 86)
        result_mul_537907 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 29), '*', h2_537905, k2_537906)
        
        # Applying the binary operator '-' (line 86)
        result_sub_537908 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 27), '-', result_add_537904, result_mul_537907)
        
        # Processing the call keyword arguments (line 86)
        kwargs_537909 = {}
        # Getting the type of 'sqrt' (line 86)
        sqrt_537897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 86)
        sqrt_call_result_537910 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), sqrt_537897, *[result_sub_537908], **kwargs_537909)
        
        int_537911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 37), 'int')
        # Getting the type of 'h2' (line 86)
        h2_537912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'h2')
        int_537913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 45), 'int')
        # Applying the binary operator '**' (line 86)
        result_pow_537914 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 41), '**', h2_537912, int_537913)
        
        # Getting the type of 'k2' (line 86)
        k2_537915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 49), 'k2')
        int_537916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 53), 'int')
        # Applying the binary operator '**' (line 86)
        result_pow_537917 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 49), '**', k2_537915, int_537916)
        
        # Applying the binary operator '+' (line 86)
        result_add_537918 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 41), '+', result_pow_537914, result_pow_537917)
        
        # Applying the binary operator '*' (line 86)
        result_mul_537919 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 37), '*', int_537911, result_add_537918)
        
        int_537920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 58), 'int')
        # Getting the type of 'h2' (line 86)
        h2_537921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'h2')
        # Applying the binary operator '*' (line 86)
        result_mul_537922 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 58), '*', int_537920, h2_537921)
        
        # Getting the type of 'k2' (line 86)
        k2_537923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 63), 'k2')
        # Applying the binary operator '*' (line 86)
        result_mul_537924 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 62), '*', result_mul_537922, k2_537923)
        
        # Getting the type of 'h2' (line 86)
        h2_537925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 67), 'h2')
        # Getting the type of 'k2' (line 86)
        k2_537926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 72), 'k2')
        # Applying the binary operator '+' (line 86)
        result_add_537927 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 67), '+', h2_537925, k2_537926)
        
        # Applying the binary operator '*' (line 86)
        result_mul_537928 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 65), '*', result_mul_537924, result_add_537927)
        
        # Applying the binary operator '+' (line 86)
        result_add_537929 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 37), '+', result_mul_537919, result_mul_537928)
        
        # Applying the binary operator '*' (line 86)
        result_mul_537930 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 8), '*', sqrt_call_result_537910, result_add_537929)
        
        # Applying the binary operator '+' (line 85)
        result_add_537931 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 75), '+', result_add_537896, result_mul_537930)
        
        # Assigning a type to the variable 'res' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'res', result_add_537931)
        int_537932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
        # Getting the type of 'pi' (line 87)
        pi_537933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'pi')
        # Applying the binary operator '*' (line 87)
        result_mul_537934 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '*', int_537932, pi_537933)
        
        int_537935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 21), 'int')
        # Applying the binary operator 'div' (line 87)
        result_div_537936 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 20), 'div', result_mul_537934, int_537935)
        
        # Getting the type of 'res' (line 87)
        res_537937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'res')
        # Applying the binary operator '*' (line 87)
        result_mul_537938 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '*', result_div_537936, res_537937)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', result_mul_537938)
        
        # ################# End of 'G22(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G22' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_537939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_537939)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G22'
        return stypy_return_type_537939

    # Assigning a type to the variable 'G22' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'G22', G22)

    @norecursion
    def G21(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G21'
        module_type_store = module_type_store.open_function_context('G21', 89, 4, False)
        
        # Passed parameters checking function
        G21.stypy_localization = localization
        G21.stypy_type_of_self = None
        G21.stypy_type_store = module_type_store
        G21.stypy_function_name = 'G21'
        G21.stypy_param_names_list = ['h2', 'k2']
        G21.stypy_varargs_param_name = None
        G21.stypy_kwargs_param_name = None
        G21.stypy_call_defaults = defaults
        G21.stypy_call_varargs = varargs
        G21.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G21', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G21', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G21(...)' code ##################

        
        # Assigning a BinOp to a Name (line 90):
        
        # Assigning a BinOp to a Name (line 90):
        int_537940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'int')
        # Getting the type of 'h2' (line 90)
        h2_537941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'h2')
        int_537942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537943 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 18), '**', h2_537941, int_537942)
        
        # Getting the type of 'k2' (line 90)
        k2_537944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'k2')
        int_537945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537946 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 26), '**', k2_537944, int_537945)
        
        # Applying the binary operator '+' (line 90)
        result_add_537947 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 18), '+', result_pow_537943, result_pow_537946)
        
        # Applying the binary operator '*' (line 90)
        result_mul_537948 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '*', int_537940, result_add_537947)
        
        int_537949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'int')
        # Getting the type of 'h2' (line 90)
        h2_537950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 37), 'h2')
        # Applying the binary operator '*' (line 90)
        result_mul_537951 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 35), '*', int_537949, h2_537950)
        
        # Getting the type of 'k2' (line 90)
        k2_537952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'k2')
        # Applying the binary operator '*' (line 90)
        result_mul_537953 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 39), '*', result_mul_537951, k2_537952)
        
        # Getting the type of 'h2' (line 90)
        h2_537954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 44), 'h2')
        int_537955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 48), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537956 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 44), '**', h2_537954, int_537955)
        
        # Getting the type of 'k2' (line 90)
        k2_537957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 52), 'k2')
        int_537958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 56), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537959 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 52), '**', k2_537957, int_537958)
        
        # Applying the binary operator '+' (line 90)
        result_add_537960 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 44), '+', result_pow_537956, result_pow_537959)
        
        # Applying the binary operator '*' (line 90)
        result_mul_537961 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 42), '*', result_mul_537953, result_add_537960)
        
        # Applying the binary operator '-' (line 90)
        result_sub_537962 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '-', result_mul_537948, result_mul_537961)
        
        int_537963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 61), 'int')
        # Getting the type of 'h2' (line 90)
        h2_537964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 63), 'h2')
        int_537965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 67), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537966 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 63), '**', h2_537964, int_537965)
        
        # Applying the binary operator '*' (line 90)
        result_mul_537967 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 61), '*', int_537963, result_pow_537966)
        
        # Getting the type of 'k2' (line 90)
        k2_537968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 69), 'k2')
        int_537969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 73), 'int')
        # Applying the binary operator '**' (line 90)
        result_pow_537970 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 69), '**', k2_537968, int_537969)
        
        # Applying the binary operator '*' (line 90)
        result_mul_537971 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 68), '*', result_mul_537967, result_pow_537970)
        
        # Applying the binary operator '+' (line 90)
        result_add_537972 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 59), '+', result_sub_537962, result_mul_537971)
        
        
        # Call to sqrt(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'h2' (line 91)
        h2_537974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'h2', False)
        int_537975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 19), 'int')
        # Applying the binary operator '**' (line 91)
        result_pow_537976 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), '**', h2_537974, int_537975)
        
        # Getting the type of 'k2' (line 91)
        k2_537977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'k2', False)
        int_537978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
        # Applying the binary operator '**' (line 91)
        result_pow_537979 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 23), '**', k2_537977, int_537978)
        
        # Applying the binary operator '+' (line 91)
        result_add_537980 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), '+', result_pow_537976, result_pow_537979)
        
        # Getting the type of 'h2' (line 91)
        h2_537981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'h2', False)
        # Getting the type of 'k2' (line 91)
        k2_537982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'k2', False)
        # Applying the binary operator '*' (line 91)
        result_mul_537983 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 31), '*', h2_537981, k2_537982)
        
        # Applying the binary operator '-' (line 91)
        result_sub_537984 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 29), '-', result_add_537980, result_mul_537983)
        
        # Processing the call keyword arguments (line 91)
        kwargs_537985 = {}
        # Getting the type of 'sqrt' (line 91)
        sqrt_537973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 91)
        sqrt_call_result_537986 = invoke(stypy.reporting.localization.Localization(__file__, 91, 10), sqrt_537973, *[result_sub_537984], **kwargs_537985)
        
        int_537987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'int')
        # Getting the type of 'h2' (line 91)
        h2_537988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'h2')
        int_537989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 46), 'int')
        # Applying the binary operator '**' (line 91)
        result_pow_537990 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 42), '**', h2_537988, int_537989)
        
        # Getting the type of 'k2' (line 91)
        k2_537991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 50), 'k2')
        int_537992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 54), 'int')
        # Applying the binary operator '**' (line 91)
        result_pow_537993 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 50), '**', k2_537991, int_537992)
        
        # Applying the binary operator '+' (line 91)
        result_add_537994 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 42), '+', result_pow_537990, result_pow_537993)
        
        # Applying the binary operator '*' (line 91)
        result_mul_537995 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 39), '*', int_537987, result_add_537994)
        
        int_537996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 59), 'int')
        # Getting the type of 'h2' (line 91)
        h2_537997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 61), 'h2')
        # Applying the binary operator '*' (line 91)
        result_mul_537998 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 59), '*', int_537996, h2_537997)
        
        # Getting the type of 'k2' (line 91)
        k2_537999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'k2')
        # Applying the binary operator '*' (line 91)
        result_mul_538000 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 63), '*', result_mul_537998, k2_537999)
        
        # Getting the type of 'h2' (line 91)
        h2_538001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'h2')
        # Getting the type of 'k2' (line 91)
        k2_538002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 73), 'k2')
        # Applying the binary operator '+' (line 91)
        result_add_538003 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 68), '+', h2_538001, k2_538002)
        
        # Applying the binary operator '*' (line 91)
        result_mul_538004 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 66), '*', result_mul_538000, result_add_538003)
        
        # Applying the binary operator '-' (line 91)
        result_sub_538005 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 39), '-', result_mul_537995, result_mul_538004)
        
        # Applying the binary operator '*' (line 91)
        result_mul_538006 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 10), '*', sqrt_call_result_537986, result_sub_538005)
        
        # Applying the binary operator '+' (line 91)
        result_add_538007 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 8), '+', result_add_537972, result_mul_538006)
        
        # Assigning a type to the variable 'res' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'res', result_add_538007)
        int_538008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 15), 'int')
        # Getting the type of 'pi' (line 92)
        pi_538009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'pi')
        # Applying the binary operator '*' (line 92)
        result_mul_538010 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '*', int_538008, pi_538009)
        
        int_538011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'int')
        # Applying the binary operator 'div' (line 92)
        result_div_538012 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 20), 'div', result_mul_538010, int_538011)
        
        # Getting the type of 'res' (line 92)
        res_538013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'res')
        # Applying the binary operator '*' (line 92)
        result_mul_538014 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 24), '*', result_div_538012, res_538013)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', result_mul_538014)
        
        # ################# End of 'G21(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G21' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_538015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538015)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G21'
        return stypy_return_type_538015

    # Assigning a type to the variable 'G21' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'G21', G21)

    @norecursion
    def G23(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G23'
        module_type_store = module_type_store.open_function_context('G23', 94, 4, False)
        
        # Passed parameters checking function
        G23.stypy_localization = localization
        G23.stypy_type_of_self = None
        G23.stypy_type_store = module_type_store
        G23.stypy_function_name = 'G23'
        G23.stypy_param_names_list = ['h2', 'k2']
        G23.stypy_varargs_param_name = None
        G23.stypy_kwargs_param_name = None
        G23.stypy_call_defaults = defaults
        G23.stypy_call_varargs = varargs
        G23.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G23', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G23', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G23(...)' code ##################

        int_538016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 15), 'int')
        # Getting the type of 'pi' (line 95)
        pi_538017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'pi')
        # Applying the binary operator '*' (line 95)
        result_mul_538018 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), '*', int_538016, pi_538017)
        
        # Getting the type of 'h2' (line 95)
        h2_538019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'h2')
        int_538020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'int')
        # Applying the binary operator '**' (line 95)
        result_pow_538021 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 20), '**', h2_538019, int_538020)
        
        # Applying the binary operator '*' (line 95)
        result_mul_538022 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 19), '*', result_mul_538018, result_pow_538021)
        
        # Getting the type of 'k2' (line 95)
        k2_538023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'k2')
        # Applying the binary operator '*' (line 95)
        result_mul_538024 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 25), '*', result_mul_538022, k2_538023)
        
        # Getting the type of 'k2' (line 95)
        k2_538025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'k2')
        # Getting the type of 'h2' (line 95)
        h2_538026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'h2')
        # Applying the binary operator '-' (line 95)
        result_sub_538027 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 30), '-', k2_538025, h2_538026)
        
        # Applying the binary operator '*' (line 95)
        result_mul_538028 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '*', result_mul_538024, result_sub_538027)
        
        int_538029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'int')
        # Applying the binary operator 'div' (line 95)
        result_div_538030 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 38), 'div', result_mul_538028, int_538029)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', result_div_538030)
        
        # ################# End of 'G23(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G23' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_538031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538031)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G23'
        return stypy_return_type_538031

    # Assigning a type to the variable 'G23' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'G23', G23)

    @norecursion
    def G24(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G24'
        module_type_store = module_type_store.open_function_context('G24', 97, 4, False)
        
        # Passed parameters checking function
        G24.stypy_localization = localization
        G24.stypy_type_of_self = None
        G24.stypy_type_store = module_type_store
        G24.stypy_function_name = 'G24'
        G24.stypy_param_names_list = ['h2', 'k2']
        G24.stypy_varargs_param_name = None
        G24.stypy_kwargs_param_name = None
        G24.stypy_call_defaults = defaults
        G24.stypy_call_varargs = varargs
        G24.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G24', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G24', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G24(...)' code ##################

        int_538032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 15), 'int')
        # Getting the type of 'pi' (line 98)
        pi_538033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'pi')
        # Applying the binary operator '*' (line 98)
        result_mul_538034 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 15), '*', int_538032, pi_538033)
        
        # Getting the type of 'h2' (line 98)
        h2_538035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'h2')
        # Applying the binary operator '*' (line 98)
        result_mul_538036 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 19), '*', result_mul_538034, h2_538035)
        
        # Getting the type of 'k2' (line 98)
        k2_538037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'k2')
        int_538038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 27), 'int')
        # Applying the binary operator '**' (line 98)
        result_pow_538039 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 23), '**', k2_538037, int_538038)
        
        # Applying the binary operator '*' (line 98)
        result_mul_538040 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 22), '*', result_mul_538036, result_pow_538039)
        
        # Getting the type of 'k2' (line 98)
        k2_538041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'k2')
        # Getting the type of 'h2' (line 98)
        h2_538042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'h2')
        # Applying the binary operator '-' (line 98)
        result_sub_538043 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 30), '-', k2_538041, h2_538042)
        
        # Applying the binary operator '*' (line 98)
        result_mul_538044 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 28), '*', result_mul_538040, result_sub_538043)
        
        int_538045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 39), 'int')
        # Applying the binary operator 'div' (line 98)
        result_div_538046 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 38), 'div', result_mul_538044, int_538045)
        
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', result_div_538046)
        
        # ################# End of 'G24(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G24' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_538047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538047)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G24'
        return stypy_return_type_538047

    # Assigning a type to the variable 'G24' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'G24', G24)

    @norecursion
    def G25(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G25'
        module_type_store = module_type_store.open_function_context('G25', 100, 4, False)
        
        # Passed parameters checking function
        G25.stypy_localization = localization
        G25.stypy_type_of_self = None
        G25.stypy_type_store = module_type_store
        G25.stypy_function_name = 'G25'
        G25.stypy_param_names_list = ['h2', 'k2']
        G25.stypy_varargs_param_name = None
        G25.stypy_kwargs_param_name = None
        G25.stypy_call_defaults = defaults
        G25.stypy_call_varargs = varargs
        G25.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G25', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G25', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G25(...)' code ##################

        int_538048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'int')
        # Getting the type of 'pi' (line 101)
        pi_538049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'pi')
        # Applying the binary operator '*' (line 101)
        result_mul_538050 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '*', int_538048, pi_538049)
        
        # Getting the type of 'h2' (line 101)
        h2_538051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'h2')
        # Applying the binary operator '*' (line 101)
        result_mul_538052 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 19), '*', result_mul_538050, h2_538051)
        
        # Getting the type of 'k2' (line 101)
        k2_538053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'k2')
        # Applying the binary operator '*' (line 101)
        result_mul_538054 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 22), '*', result_mul_538052, k2_538053)
        
        # Getting the type of 'k2' (line 101)
        k2_538055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'k2')
        # Getting the type of 'h2' (line 101)
        h2_538056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'h2')
        # Applying the binary operator '-' (line 101)
        result_sub_538057 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 27), '-', k2_538055, h2_538056)
        
        int_538058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'int')
        # Applying the binary operator '**' (line 101)
        result_pow_538059 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 26), '**', result_sub_538057, int_538058)
        
        # Applying the binary operator '*' (line 101)
        result_mul_538060 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 25), '*', result_mul_538054, result_pow_538059)
        
        int_538061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 39), 'int')
        # Applying the binary operator 'div' (line 101)
        result_div_538062 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 38), 'div', result_mul_538060, int_538061)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', result_div_538062)
        
        # ################# End of 'G25(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G25' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_538063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G25'
        return stypy_return_type_538063

    # Assigning a type to the variable 'G25' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'G25', G25)

    @norecursion
    def G32(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G32'
        module_type_store = module_type_store.open_function_context('G32', 103, 4, False)
        
        # Passed parameters checking function
        G32.stypy_localization = localization
        G32.stypy_type_of_self = None
        G32.stypy_type_store = module_type_store
        G32.stypy_function_name = 'G32'
        G32.stypy_param_names_list = ['h2', 'k2']
        G32.stypy_varargs_param_name = None
        G32.stypy_kwargs_param_name = None
        G32.stypy_call_defaults = defaults
        G32.stypy_call_varargs = varargs
        G32.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G32', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G32', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G32(...)' code ##################

        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        int_538064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 15), 'int')
        # Getting the type of 'h2' (line 104)
        h2_538065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'h2')
        int_538066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538067 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 19), '**', h2_538065, int_538066)
        
        # Getting the type of 'k2' (line 104)
        k2_538068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'k2')
        int_538069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538070 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 27), '**', k2_538068, int_538069)
        
        # Applying the binary operator '+' (line 104)
        result_add_538071 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 19), '+', result_pow_538067, result_pow_538070)
        
        # Applying the binary operator '*' (line 104)
        result_mul_538072 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '*', int_538064, result_add_538071)
        
        int_538073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'int')
        # Getting the type of 'h2' (line 104)
        h2_538074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'h2')
        # Applying the binary operator '*' (line 104)
        result_mul_538075 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 36), '*', int_538073, h2_538074)
        
        # Getting the type of 'k2' (line 104)
        k2_538076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 42), 'k2')
        # Applying the binary operator '*' (line 104)
        result_mul_538077 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 41), '*', result_mul_538075, k2_538076)
        
        # Getting the type of 'h2' (line 104)
        h2_538078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 46), 'h2')
        int_538079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 50), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538080 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 46), '**', h2_538078, int_538079)
        
        # Getting the type of 'k2' (line 104)
        k2_538081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 54), 'k2')
        int_538082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 58), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538083 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 54), '**', k2_538081, int_538082)
        
        # Applying the binary operator '+' (line 104)
        result_add_538084 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 46), '+', result_pow_538080, result_pow_538083)
        
        # Applying the binary operator '*' (line 104)
        result_mul_538085 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 44), '*', result_mul_538077, result_add_538084)
        
        # Applying the binary operator '-' (line 104)
        result_sub_538086 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), '-', result_mul_538072, result_mul_538085)
        
        int_538087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 63), 'int')
        # Getting the type of 'h2' (line 104)
        h2_538088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 66), 'h2')
        int_538089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 70), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538090 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 66), '**', h2_538088, int_538089)
        
        # Applying the binary operator '*' (line 104)
        result_mul_538091 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 63), '*', int_538087, result_pow_538090)
        
        # Getting the type of 'k2' (line 104)
        k2_538092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 72), 'k2')
        int_538093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 76), 'int')
        # Applying the binary operator '**' (line 104)
        result_pow_538094 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 72), '**', k2_538092, int_538093)
        
        # Applying the binary operator '*' (line 104)
        result_mul_538095 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 71), '*', result_mul_538091, result_pow_538094)
        
        # Applying the binary operator '+' (line 104)
        result_add_538096 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 61), '+', result_sub_538086, result_mul_538095)
        
        
        # Call to sqrt(...): (line 105)
        # Processing the call arguments (line 105)
        int_538098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'int')
        # Getting the type of 'h2' (line 105)
        h2_538099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'h2', False)
        int_538100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'int')
        # Applying the binary operator '**' (line 105)
        result_pow_538101 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 18), '**', h2_538099, int_538100)
        
        # Getting the type of 'k2' (line 105)
        k2_538102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'k2', False)
        int_538103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 30), 'int')
        # Applying the binary operator '**' (line 105)
        result_pow_538104 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 26), '**', k2_538102, int_538103)
        
        # Applying the binary operator '+' (line 105)
        result_add_538105 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 18), '+', result_pow_538101, result_pow_538104)
        
        # Applying the binary operator '*' (line 105)
        result_mul_538106 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), '*', int_538098, result_add_538105)
        
        int_538107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 35), 'int')
        # Getting the type of 'h2' (line 105)
        h2_538108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 'h2', False)
        # Applying the binary operator '*' (line 105)
        result_mul_538109 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 35), '*', int_538107, h2_538108)
        
        # Getting the type of 'k2' (line 105)
        k2_538110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'k2', False)
        # Applying the binary operator '*' (line 105)
        result_mul_538111 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 39), '*', result_mul_538109, k2_538110)
        
        # Applying the binary operator '-' (line 105)
        result_sub_538112 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), '-', result_mul_538106, result_mul_538111)
        
        # Processing the call keyword arguments (line 105)
        kwargs_538113 = {}
        # Getting the type of 'sqrt' (line 105)
        sqrt_538097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 105)
        sqrt_call_result_538114 = invoke(stypy.reporting.localization.Localization(__file__, 105, 10), sqrt_538097, *[result_sub_538112], **kwargs_538113)
        
        int_538115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'int')
        # Getting the type of 'h2' (line 105)
        h2_538116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 49), 'h2')
        int_538117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 53), 'int')
        # Applying the binary operator '**' (line 105)
        result_pow_538118 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 49), '**', h2_538116, int_538117)
        
        # Getting the type of 'k2' (line 105)
        k2_538119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 57), 'k2')
        int_538120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 61), 'int')
        # Applying the binary operator '**' (line 105)
        result_pow_538121 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 57), '**', k2_538119, int_538120)
        
        # Applying the binary operator '+' (line 105)
        result_add_538122 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 49), '+', result_pow_538118, result_pow_538121)
        
        # Applying the binary operator '*' (line 105)
        result_mul_538123 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 45), '*', int_538115, result_add_538122)
        
        int_538124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Getting the type of 'h2' (line 106)
        h2_538125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'h2')
        # Applying the binary operator '*' (line 106)
        result_mul_538126 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 8), '*', int_538124, h2_538125)
        
        # Getting the type of 'k2' (line 106)
        k2_538127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'k2')
        # Applying the binary operator '*' (line 106)
        result_mul_538128 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 13), '*', result_mul_538126, k2_538127)
        
        # Getting the type of 'h2' (line 106)
        h2_538129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'h2')
        # Getting the type of 'k2' (line 106)
        k2_538130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'k2')
        # Applying the binary operator '+' (line 106)
        result_add_538131 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 18), '+', h2_538129, k2_538130)
        
        # Applying the binary operator '*' (line 106)
        result_mul_538132 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 16), '*', result_mul_538128, result_add_538131)
        
        # Applying the binary operator '+' (line 105)
        result_add_538133 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 45), '+', result_mul_538123, result_mul_538132)
        
        # Applying the binary operator '*' (line 105)
        result_mul_538134 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 10), '*', sqrt_call_result_538114, result_add_538133)
        
        # Applying the binary operator '+' (line 105)
        result_add_538135 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 8), '+', result_add_538096, result_mul_538134)
        
        # Assigning a type to the variable 'res' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'res', result_add_538135)
        int_538136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'int')
        # Getting the type of 'pi' (line 107)
        pi_538137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'pi')
        # Applying the binary operator '*' (line 107)
        result_mul_538138 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '*', int_538136, pi_538137)
        
        int_538139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'int')
        # Applying the binary operator 'div' (line 107)
        result_div_538140 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 20), 'div', result_mul_538138, int_538139)
        
        # Getting the type of 'k2' (line 107)
        k2_538141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 27), 'k2')
        # Applying the binary operator '*' (line 107)
        result_mul_538142 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 26), '*', result_div_538140, k2_538141)
        
        # Getting the type of 'h2' (line 107)
        h2_538143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'h2')
        # Applying the binary operator '*' (line 107)
        result_mul_538144 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '*', result_mul_538142, h2_538143)
        
        # Getting the type of 'res' (line 107)
        res_538145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'res')
        # Applying the binary operator '*' (line 107)
        result_mul_538146 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 32), '*', result_mul_538144, res_538145)
        
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', result_mul_538146)
        
        # ################# End of 'G32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G32' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_538147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538147)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G32'
        return stypy_return_type_538147

    # Assigning a type to the variable 'G32' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'G32', G32)

    @norecursion
    def G31(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G31'
        module_type_store = module_type_store.open_function_context('G31', 109, 4, False)
        
        # Passed parameters checking function
        G31.stypy_localization = localization
        G31.stypy_type_of_self = None
        G31.stypy_type_store = module_type_store
        G31.stypy_function_name = 'G31'
        G31.stypy_param_names_list = ['h2', 'k2']
        G31.stypy_varargs_param_name = None
        G31.stypy_kwargs_param_name = None
        G31.stypy_call_defaults = defaults
        G31.stypy_call_varargs = varargs
        G31.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G31', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G31', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G31(...)' code ##################

        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        int_538148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
        # Getting the type of 'h2' (line 110)
        h2_538149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'h2')
        int_538150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 23), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538151 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 19), '**', h2_538149, int_538150)
        
        # Getting the type of 'k2' (line 110)
        k2_538152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'k2')
        int_538153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538154 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 27), '**', k2_538152, int_538153)
        
        # Applying the binary operator '+' (line 110)
        result_add_538155 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 19), '+', result_pow_538151, result_pow_538154)
        
        # Applying the binary operator '*' (line 110)
        result_mul_538156 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '*', int_538148, result_add_538155)
        
        int_538157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 36), 'int')
        # Getting the type of 'h2' (line 110)
        h2_538158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'h2')
        # Applying the binary operator '*' (line 110)
        result_mul_538159 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 36), '*', int_538157, h2_538158)
        
        # Getting the type of 'k2' (line 110)
        k2_538160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 42), 'k2')
        # Applying the binary operator '*' (line 110)
        result_mul_538161 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 41), '*', result_mul_538159, k2_538160)
        
        # Getting the type of 'h2' (line 110)
        h2_538162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 46), 'h2')
        int_538163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 50), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538164 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 46), '**', h2_538162, int_538163)
        
        # Getting the type of 'k2' (line 110)
        k2_538165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 54), 'k2')
        int_538166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 58), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538167 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 54), '**', k2_538165, int_538166)
        
        # Applying the binary operator '+' (line 110)
        result_add_538168 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 46), '+', result_pow_538164, result_pow_538167)
        
        # Applying the binary operator '*' (line 110)
        result_mul_538169 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 44), '*', result_mul_538161, result_add_538168)
        
        # Applying the binary operator '-' (line 110)
        result_sub_538170 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '-', result_mul_538156, result_mul_538169)
        
        int_538171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'int')
        # Getting the type of 'h2' (line 110)
        h2_538172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 66), 'h2')
        int_538173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 70), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538174 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 66), '**', h2_538172, int_538173)
        
        # Applying the binary operator '*' (line 110)
        result_mul_538175 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 63), '*', int_538171, result_pow_538174)
        
        # Getting the type of 'k2' (line 110)
        k2_538176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 72), 'k2')
        int_538177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 76), 'int')
        # Applying the binary operator '**' (line 110)
        result_pow_538178 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 72), '**', k2_538176, int_538177)
        
        # Applying the binary operator '*' (line 110)
        result_mul_538179 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 71), '*', result_mul_538175, result_pow_538178)
        
        # Applying the binary operator '+' (line 110)
        result_add_538180 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 61), '+', result_sub_538170, result_mul_538179)
        
        
        # Call to sqrt(...): (line 111)
        # Processing the call arguments (line 111)
        int_538182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 15), 'int')
        # Getting the type of 'h2' (line 111)
        h2_538183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'h2', False)
        int_538184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 22), 'int')
        # Applying the binary operator '**' (line 111)
        result_pow_538185 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 18), '**', h2_538183, int_538184)
        
        # Getting the type of 'k2' (line 111)
        k2_538186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'k2', False)
        int_538187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 30), 'int')
        # Applying the binary operator '**' (line 111)
        result_pow_538188 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 26), '**', k2_538186, int_538187)
        
        # Applying the binary operator '+' (line 111)
        result_add_538189 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 18), '+', result_pow_538185, result_pow_538188)
        
        # Applying the binary operator '*' (line 111)
        result_mul_538190 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '*', int_538182, result_add_538189)
        
        int_538191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'int')
        # Getting the type of 'h2' (line 111)
        h2_538192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'h2', False)
        # Applying the binary operator '*' (line 111)
        result_mul_538193 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 35), '*', int_538191, h2_538192)
        
        # Getting the type of 'k2' (line 111)
        k2_538194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'k2', False)
        # Applying the binary operator '*' (line 111)
        result_mul_538195 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 39), '*', result_mul_538193, k2_538194)
        
        # Applying the binary operator '-' (line 111)
        result_sub_538196 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), '-', result_mul_538190, result_mul_538195)
        
        # Processing the call keyword arguments (line 111)
        kwargs_538197 = {}
        # Getting the type of 'sqrt' (line 111)
        sqrt_538181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 111)
        sqrt_call_result_538198 = invoke(stypy.reporting.localization.Localization(__file__, 111, 10), sqrt_538181, *[result_sub_538196], **kwargs_538197)
        
        int_538199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 45), 'int')
        # Getting the type of 'h2' (line 111)
        h2_538200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 48), 'h2')
        int_538201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 52), 'int')
        # Applying the binary operator '**' (line 111)
        result_pow_538202 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 48), '**', h2_538200, int_538201)
        
        # Getting the type of 'k2' (line 111)
        k2_538203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 56), 'k2')
        int_538204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 60), 'int')
        # Applying the binary operator '**' (line 111)
        result_pow_538205 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 56), '**', k2_538203, int_538204)
        
        # Applying the binary operator '+' (line 111)
        result_add_538206 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 48), '+', result_pow_538202, result_pow_538205)
        
        # Applying the binary operator '*' (line 111)
        result_mul_538207 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 45), '*', int_538199, result_add_538206)
        
        int_538208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'int')
        # Getting the type of 'h2' (line 112)
        h2_538209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'h2')
        # Applying the binary operator '*' (line 112)
        result_mul_538210 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 8), '*', int_538208, h2_538209)
        
        # Getting the type of 'k2' (line 112)
        k2_538211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'k2')
        # Applying the binary operator '*' (line 112)
        result_mul_538212 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 13), '*', result_mul_538210, k2_538211)
        
        # Getting the type of 'h2' (line 112)
        h2_538213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'h2')
        # Getting the type of 'k2' (line 112)
        k2_538214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'k2')
        # Applying the binary operator '+' (line 112)
        result_add_538215 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 18), '+', h2_538213, k2_538214)
        
        # Applying the binary operator '*' (line 112)
        result_mul_538216 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 16), '*', result_mul_538212, result_add_538215)
        
        # Applying the binary operator '-' (line 111)
        result_sub_538217 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 45), '-', result_mul_538207, result_mul_538216)
        
        # Applying the binary operator '*' (line 111)
        result_mul_538218 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 10), '*', sqrt_call_result_538198, result_sub_538217)
        
        # Applying the binary operator '+' (line 111)
        result_add_538219 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 8), '+', result_add_538180, result_mul_538218)
        
        # Assigning a type to the variable 'res' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'res', result_add_538219)
        int_538220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'int')
        # Getting the type of 'pi' (line 113)
        pi_538221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'pi')
        # Applying the binary operator '*' (line 113)
        result_mul_538222 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), '*', int_538220, pi_538221)
        
        int_538223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 21), 'int')
        # Applying the binary operator 'div' (line 113)
        result_div_538224 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 20), 'div', result_mul_538222, int_538223)
        
        # Getting the type of 'h2' (line 113)
        h2_538225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'h2')
        # Applying the binary operator '*' (line 113)
        result_mul_538226 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 26), '*', result_div_538224, h2_538225)
        
        # Getting the type of 'k2' (line 113)
        k2_538227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 30), 'k2')
        # Applying the binary operator '*' (line 113)
        result_mul_538228 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 29), '*', result_mul_538226, k2_538227)
        
        # Getting the type of 'res' (line 113)
        res_538229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'res')
        # Applying the binary operator '*' (line 113)
        result_mul_538230 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 32), '*', result_mul_538228, res_538229)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', result_mul_538230)
        
        # ################# End of 'G31(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G31' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_538231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538231)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G31'
        return stypy_return_type_538231

    # Assigning a type to the variable 'G31' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'G31', G31)

    @norecursion
    def G34(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G34'
        module_type_store = module_type_store.open_function_context('G34', 115, 4, False)
        
        # Passed parameters checking function
        G34.stypy_localization = localization
        G34.stypy_type_of_self = None
        G34.stypy_type_store = module_type_store
        G34.stypy_function_name = 'G34'
        G34.stypy_param_names_list = ['h2', 'k2']
        G34.stypy_varargs_param_name = None
        G34.stypy_kwargs_param_name = None
        G34.stypy_call_defaults = defaults
        G34.stypy_call_varargs = varargs
        G34.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G34', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G34', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G34(...)' code ##################

        
        # Assigning a BinOp to a Name (line 116):
        
        # Assigning a BinOp to a Name (line 116):
        int_538232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'int')
        # Getting the type of 'h2' (line 116)
        h2_538233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'h2')
        int_538234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538235 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '**', h2_538233, int_538234)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538236 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '*', int_538232, result_pow_538235)
        
        int_538237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'int')
        # Getting the type of 'k2' (line 116)
        k2_538238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'k2')
        int_538239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538240 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 28), '**', k2_538238, int_538239)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538241 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 25), '*', int_538237, result_pow_538240)
        
        # Applying the binary operator '+' (line 116)
        result_add_538242 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '+', result_mul_538236, result_mul_538241)
        
        int_538243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'int')
        # Getting the type of 'h2' (line 116)
        h2_538244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), 'h2')
        int_538245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 43), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538246 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 39), '**', h2_538244, int_538245)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538247 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 36), '*', int_538243, result_pow_538246)
        
        # Getting the type of 'k2' (line 116)
        k2_538248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 45), 'k2')
        # Applying the binary operator '*' (line 116)
        result_mul_538249 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 44), '*', result_mul_538247, k2_538248)
        
        # Applying the binary operator '-' (line 116)
        result_sub_538250 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 34), '-', result_add_538242, result_mul_538249)
        
        int_538251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 50), 'int')
        # Getting the type of 'h2' (line 116)
        h2_538252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 53), 'h2')
        # Applying the binary operator '*' (line 116)
        result_mul_538253 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 50), '*', int_538251, h2_538252)
        
        # Getting the type of 'k2' (line 116)
        k2_538254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 56), 'k2')
        int_538255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538256 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 56), '**', k2_538254, int_538255)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538257 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 55), '*', result_mul_538253, result_pow_538256)
        
        # Applying the binary operator '-' (line 116)
        result_sub_538258 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 48), '-', result_sub_538250, result_mul_538257)
        
        int_538259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 64), 'int')
        # Getting the type of 'h2' (line 116)
        h2_538260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 67), 'h2')
        int_538261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 71), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538262 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 67), '**', h2_538260, int_538261)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538263 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 64), '*', int_538259, result_pow_538262)
        
        # Getting the type of 'k2' (line 116)
        k2_538264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 73), 'k2')
        int_538265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 77), 'int')
        # Applying the binary operator '**' (line 116)
        result_pow_538266 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 73), '**', k2_538264, int_538265)
        
        # Applying the binary operator '*' (line 116)
        result_mul_538267 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 72), '*', result_mul_538263, result_pow_538266)
        
        # Applying the binary operator '+' (line 116)
        result_add_538268 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 62), '+', result_sub_538258, result_mul_538267)
        
        
        # Call to sqrt(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'h2' (line 117)
        h2_538270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'h2', False)
        int_538271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 19), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_538272 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 15), '**', h2_538270, int_538271)
        
        int_538273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'int')
        # Getting the type of 'k2' (line 117)
        k2_538274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'k2', False)
        int_538275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_538276 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 25), '**', k2_538274, int_538275)
        
        # Applying the binary operator '*' (line 117)
        result_mul_538277 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 23), '*', int_538273, result_pow_538276)
        
        # Applying the binary operator '+' (line 117)
        result_add_538278 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 15), '+', result_pow_538272, result_mul_538277)
        
        # Getting the type of 'h2' (line 117)
        h2_538279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'h2', False)
        # Getting the type of 'k2' (line 117)
        k2_538280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'k2', False)
        # Applying the binary operator '*' (line 117)
        result_mul_538281 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 33), '*', h2_538279, k2_538280)
        
        # Applying the binary operator '-' (line 117)
        result_sub_538282 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 31), '-', result_add_538278, result_mul_538281)
        
        # Processing the call keyword arguments (line 117)
        kwargs_538283 = {}
        # Getting the type of 'sqrt' (line 117)
        sqrt_538269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 117)
        sqrt_call_result_538284 = invoke(stypy.reporting.localization.Localization(__file__, 117, 10), sqrt_538269, *[result_sub_538282], **kwargs_538283)
        
        int_538285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 41), 'int')
        # Getting the type of 'h2' (line 117)
        h2_538286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'h2')
        int_538287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 48), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_538288 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 44), '**', h2_538286, int_538287)
        
        # Applying the binary operator '*' (line 117)
        result_mul_538289 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 41), '*', int_538285, result_pow_538288)
        
        int_538290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 52), 'int')
        # Getting the type of 'k2' (line 117)
        k2_538291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 54), 'k2')
        int_538292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 58), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_538293 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 54), '**', k2_538291, int_538292)
        
        # Applying the binary operator '*' (line 117)
        result_mul_538294 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 52), '*', int_538290, result_pow_538293)
        
        # Applying the binary operator '-' (line 117)
        result_sub_538295 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 41), '-', result_mul_538289, result_mul_538294)
        
        int_538296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 62), 'int')
        # Getting the type of 'h2' (line 117)
        h2_538297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 64), 'h2')
        int_538298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 68), 'int')
        # Applying the binary operator '**' (line 117)
        result_pow_538299 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 64), '**', h2_538297, int_538298)
        
        # Applying the binary operator '*' (line 117)
        result_mul_538300 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 62), '*', int_538296, result_pow_538299)
        
        # Getting the type of 'k2' (line 117)
        k2_538301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 70), 'k2')
        # Applying the binary operator '*' (line 117)
        result_mul_538302 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 69), '*', result_mul_538300, k2_538301)
        
        # Applying the binary operator '+' (line 117)
        result_add_538303 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 60), '+', result_sub_538295, result_mul_538302)
        
        int_538304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'int')
        # Getting the type of 'h2' (line 118)
        h2_538305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'h2')
        # Applying the binary operator '*' (line 118)
        result_mul_538306 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 44), '*', int_538304, h2_538305)
        
        # Getting the type of 'k2' (line 118)
        k2_538307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 50), 'k2')
        int_538308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 54), 'int')
        # Applying the binary operator '**' (line 118)
        result_pow_538309 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 50), '**', k2_538307, int_538308)
        
        # Applying the binary operator '*' (line 118)
        result_mul_538310 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 49), '*', result_mul_538306, result_pow_538309)
        
        # Applying the binary operator '+' (line 117)
        result_add_538311 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 73), '+', result_add_538303, result_mul_538310)
        
        # Applying the binary operator '*' (line 117)
        result_mul_538312 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 10), '*', sqrt_call_result_538284, result_add_538311)
        
        # Applying the binary operator '+' (line 117)
        result_add_538313 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 8), '+', result_add_538268, result_mul_538312)
        
        # Assigning a type to the variable 'res' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'res', result_add_538313)
        int_538314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'int')
        # Getting the type of 'pi' (line 119)
        pi_538315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'pi')
        # Applying the binary operator '*' (line 119)
        result_mul_538316 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '*', int_538314, pi_538315)
        
        int_538317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
        # Applying the binary operator 'div' (line 119)
        result_div_538318 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), 'div', result_mul_538316, int_538317)
        
        # Getting the type of 'h2' (line 119)
        h2_538319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'h2')
        # Applying the binary operator '*' (line 119)
        result_mul_538320 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 26), '*', result_div_538318, h2_538319)
        
        # Getting the type of 'k2' (line 119)
        k2_538321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'k2')
        # Getting the type of 'h2' (line 119)
        h2_538322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'h2')
        # Applying the binary operator '-' (line 119)
        result_sub_538323 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 31), '-', k2_538321, h2_538322)
        
        # Applying the binary operator '*' (line 119)
        result_mul_538324 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 29), '*', result_mul_538320, result_sub_538323)
        
        # Getting the type of 'res' (line 119)
        res_538325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'res')
        # Applying the binary operator '*' (line 119)
        result_mul_538326 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 39), '*', result_mul_538324, res_538325)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', result_mul_538326)
        
        # ################# End of 'G34(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G34' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_538327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538327)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G34'
        return stypy_return_type_538327

    # Assigning a type to the variable 'G34' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'G34', G34)

    @norecursion
    def G33(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G33'
        module_type_store = module_type_store.open_function_context('G33', 121, 4, False)
        
        # Passed parameters checking function
        G33.stypy_localization = localization
        G33.stypy_type_of_self = None
        G33.stypy_type_store = module_type_store
        G33.stypy_function_name = 'G33'
        G33.stypy_param_names_list = ['h2', 'k2']
        G33.stypy_varargs_param_name = None
        G33.stypy_kwargs_param_name = None
        G33.stypy_call_defaults = defaults
        G33.stypy_call_varargs = varargs
        G33.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G33', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G33', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G33(...)' code ##################

        
        # Assigning a BinOp to a Name (line 122):
        
        # Assigning a BinOp to a Name (line 122):
        int_538328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'int')
        # Getting the type of 'h2' (line 122)
        h2_538329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'h2')
        int_538330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538331 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 17), '**', h2_538329, int_538330)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538332 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '*', int_538328, result_pow_538331)
        
        int_538333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 25), 'int')
        # Getting the type of 'k2' (line 122)
        k2_538334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'k2')
        int_538335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 32), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538336 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 28), '**', k2_538334, int_538335)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538337 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 25), '*', int_538333, result_pow_538336)
        
        # Applying the binary operator '+' (line 122)
        result_add_538338 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), '+', result_mul_538332, result_mul_538337)
        
        int_538339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 36), 'int')
        # Getting the type of 'h2' (line 122)
        h2_538340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'h2')
        int_538341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538342 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), '**', h2_538340, int_538341)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538343 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 36), '*', int_538339, result_pow_538342)
        
        # Getting the type of 'k2' (line 122)
        k2_538344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'k2')
        # Applying the binary operator '*' (line 122)
        result_mul_538345 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 44), '*', result_mul_538343, k2_538344)
        
        # Applying the binary operator '-' (line 122)
        result_sub_538346 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 34), '-', result_add_538338, result_mul_538345)
        
        int_538347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 50), 'int')
        # Getting the type of 'h2' (line 122)
        h2_538348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'h2')
        # Applying the binary operator '*' (line 122)
        result_mul_538349 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 50), '*', int_538347, h2_538348)
        
        # Getting the type of 'k2' (line 122)
        k2_538350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'k2')
        int_538351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 60), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538352 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 56), '**', k2_538350, int_538351)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538353 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 55), '*', result_mul_538349, result_pow_538352)
        
        # Applying the binary operator '-' (line 122)
        result_sub_538354 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 48), '-', result_sub_538346, result_mul_538353)
        
        int_538355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 64), 'int')
        # Getting the type of 'h2' (line 122)
        h2_538356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 67), 'h2')
        int_538357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 71), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538358 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 67), '**', h2_538356, int_538357)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538359 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 64), '*', int_538355, result_pow_538358)
        
        # Getting the type of 'k2' (line 122)
        k2_538360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 73), 'k2')
        int_538361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 77), 'int')
        # Applying the binary operator '**' (line 122)
        result_pow_538362 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 73), '**', k2_538360, int_538361)
        
        # Applying the binary operator '*' (line 122)
        result_mul_538363 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 72), '*', result_mul_538359, result_pow_538362)
        
        # Applying the binary operator '+' (line 122)
        result_add_538364 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 62), '+', result_sub_538354, result_mul_538363)
        
        
        # Call to sqrt(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'h2' (line 123)
        h2_538366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'h2', False)
        int_538367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_538368 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '**', h2_538366, int_538367)
        
        int_538369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'int')
        # Getting the type of 'k2' (line 123)
        k2_538370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'k2', False)
        int_538371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_538372 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 25), '**', k2_538370, int_538371)
        
        # Applying the binary operator '*' (line 123)
        result_mul_538373 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 23), '*', int_538369, result_pow_538372)
        
        # Applying the binary operator '+' (line 123)
        result_add_538374 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '+', result_pow_538368, result_mul_538373)
        
        # Getting the type of 'h2' (line 123)
        h2_538375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 33), 'h2', False)
        # Getting the type of 'k2' (line 123)
        k2_538376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'k2', False)
        # Applying the binary operator '*' (line 123)
        result_mul_538377 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 33), '*', h2_538375, k2_538376)
        
        # Applying the binary operator '-' (line 123)
        result_sub_538378 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 31), '-', result_add_538374, result_mul_538377)
        
        # Processing the call keyword arguments (line 123)
        kwargs_538379 = {}
        # Getting the type of 'sqrt' (line 123)
        sqrt_538365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 123)
        sqrt_call_result_538380 = invoke(stypy.reporting.localization.Localization(__file__, 123, 10), sqrt_538365, *[result_sub_538378], **kwargs_538379)
        
        int_538381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
        # Getting the type of 'h2' (line 123)
        h2_538382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'h2')
        int_538383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 47), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_538384 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '**', h2_538382, int_538383)
        
        # Applying the binary operator '*' (line 123)
        result_mul_538385 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 41), '*', int_538381, result_pow_538384)
        
        int_538386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 51), 'int')
        # Getting the type of 'k2' (line 123)
        k2_538387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 53), 'k2')
        int_538388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 57), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_538389 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 53), '**', k2_538387, int_538388)
        
        # Applying the binary operator '*' (line 123)
        result_mul_538390 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 51), '*', int_538386, result_pow_538389)
        
        # Applying the binary operator '+' (line 123)
        result_add_538391 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 41), '+', result_mul_538385, result_mul_538390)
        
        int_538392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 61), 'int')
        # Getting the type of 'h2' (line 123)
        h2_538393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'h2')
        int_538394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 67), 'int')
        # Applying the binary operator '**' (line 123)
        result_pow_538395 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 63), '**', h2_538393, int_538394)
        
        # Applying the binary operator '*' (line 123)
        result_mul_538396 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 61), '*', int_538392, result_pow_538395)
        
        # Getting the type of 'k2' (line 123)
        k2_538397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 69), 'k2')
        # Applying the binary operator '*' (line 123)
        result_mul_538398 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 68), '*', result_mul_538396, k2_538397)
        
        # Applying the binary operator '-' (line 123)
        result_sub_538399 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 59), '-', result_add_538391, result_mul_538398)
        
        int_538400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 8), 'int')
        # Getting the type of 'h2' (line 124)
        h2_538401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'h2')
        # Applying the binary operator '*' (line 124)
        result_mul_538402 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 8), '*', int_538400, h2_538401)
        
        # Getting the type of 'k2' (line 124)
        k2_538403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'k2')
        int_538404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'int')
        # Applying the binary operator '**' (line 124)
        result_pow_538405 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 14), '**', k2_538403, int_538404)
        
        # Applying the binary operator '*' (line 124)
        result_mul_538406 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), '*', result_mul_538402, result_pow_538405)
        
        # Applying the binary operator '-' (line 123)
        result_sub_538407 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 72), '-', result_sub_538399, result_mul_538406)
        
        # Applying the binary operator '*' (line 123)
        result_mul_538408 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 10), '*', sqrt_call_result_538380, result_sub_538407)
        
        # Applying the binary operator '+' (line 123)
        result_add_538409 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 8), '+', result_add_538364, result_mul_538408)
        
        # Assigning a type to the variable 'res' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'res', result_add_538409)
        int_538410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 15), 'int')
        # Getting the type of 'pi' (line 125)
        pi_538411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 18), 'pi')
        # Applying the binary operator '*' (line 125)
        result_mul_538412 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 15), '*', int_538410, pi_538411)
        
        int_538413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 21), 'int')
        # Applying the binary operator 'div' (line 125)
        result_div_538414 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 20), 'div', result_mul_538412, int_538413)
        
        # Getting the type of 'h2' (line 125)
        h2_538415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'h2')
        # Applying the binary operator '*' (line 125)
        result_mul_538416 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 26), '*', result_div_538414, h2_538415)
        
        # Getting the type of 'k2' (line 125)
        k2_538417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 31), 'k2')
        # Getting the type of 'h2' (line 125)
        h2_538418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'h2')
        # Applying the binary operator '-' (line 125)
        result_sub_538419 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 31), '-', k2_538417, h2_538418)
        
        # Applying the binary operator '*' (line 125)
        result_mul_538420 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 29), '*', result_mul_538416, result_sub_538419)
        
        # Getting the type of 'res' (line 125)
        res_538421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 40), 'res')
        # Applying the binary operator '*' (line 125)
        result_mul_538422 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 39), '*', result_mul_538420, res_538421)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', result_mul_538422)
        
        # ################# End of 'G33(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G33' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_538423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G33'
        return stypy_return_type_538423

    # Assigning a type to the variable 'G33' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'G33', G33)

    @norecursion
    def G36(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G36'
        module_type_store = module_type_store.open_function_context('G36', 127, 4, False)
        
        # Passed parameters checking function
        G36.stypy_localization = localization
        G36.stypy_type_of_self = None
        G36.stypy_type_store = module_type_store
        G36.stypy_function_name = 'G36'
        G36.stypy_param_names_list = ['h2', 'k2']
        G36.stypy_varargs_param_name = None
        G36.stypy_kwargs_param_name = None
        G36.stypy_call_defaults = defaults
        G36.stypy_call_varargs = varargs
        G36.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G36', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G36', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G36(...)' code ##################

        
        # Assigning a BinOp to a Name (line 128):
        
        # Assigning a BinOp to a Name (line 128):
        int_538424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'int')
        # Getting the type of 'h2' (line 128)
        h2_538425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'h2')
        int_538426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538427 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 18), '**', h2_538425, int_538426)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538428 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '*', int_538424, result_pow_538427)
        
        int_538429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 26), 'int')
        # Getting the type of 'k2' (line 128)
        k2_538430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'k2')
        int_538431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 32), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538432 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 28), '**', k2_538430, int_538431)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538433 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 26), '*', int_538429, result_pow_538432)
        
        # Applying the binary operator '+' (line 128)
        result_add_538434 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '+', result_mul_538428, result_mul_538433)
        
        int_538435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'int')
        # Getting the type of 'h2' (line 128)
        h2_538436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 39), 'h2')
        int_538437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 43), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538438 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 39), '**', h2_538436, int_538437)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538439 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 36), '*', int_538435, result_pow_538438)
        
        # Getting the type of 'k2' (line 128)
        k2_538440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'k2')
        # Applying the binary operator '*' (line 128)
        result_mul_538441 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 44), '*', result_mul_538439, k2_538440)
        
        # Applying the binary operator '-' (line 128)
        result_sub_538442 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 34), '-', result_add_538434, result_mul_538441)
        
        int_538443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 50), 'int')
        # Getting the type of 'h2' (line 128)
        h2_538444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 53), 'h2')
        # Applying the binary operator '*' (line 128)
        result_mul_538445 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 50), '*', int_538443, h2_538444)
        
        # Getting the type of 'k2' (line 128)
        k2_538446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 56), 'k2')
        int_538447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 60), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538448 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 56), '**', k2_538446, int_538447)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538449 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 55), '*', result_mul_538445, result_pow_538448)
        
        # Applying the binary operator '-' (line 128)
        result_sub_538450 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 48), '-', result_sub_538442, result_mul_538449)
        
        int_538451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 64), 'int')
        # Getting the type of 'h2' (line 128)
        h2_538452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 67), 'h2')
        int_538453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 71), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538454 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 67), '**', h2_538452, int_538453)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538455 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 64), '*', int_538451, result_pow_538454)
        
        # Getting the type of 'k2' (line 128)
        k2_538456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 73), 'k2')
        int_538457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 77), 'int')
        # Applying the binary operator '**' (line 128)
        result_pow_538458 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 73), '**', k2_538456, int_538457)
        
        # Applying the binary operator '*' (line 128)
        result_mul_538459 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 72), '*', result_mul_538455, result_pow_538458)
        
        # Applying the binary operator '+' (line 128)
        result_add_538460 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 62), '+', result_sub_538450, result_mul_538459)
        
        
        # Call to sqrt(...): (line 129)
        # Processing the call arguments (line 129)
        int_538462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'int')
        # Getting the type of 'h2' (line 129)
        h2_538463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'h2', False)
        int_538464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 21), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_538465 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 17), '**', h2_538463, int_538464)
        
        # Applying the binary operator '*' (line 129)
        result_mul_538466 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '*', int_538462, result_pow_538465)
        
        # Getting the type of 'k2' (line 129)
        k2_538467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'k2', False)
        int_538468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_538469 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 25), '**', k2_538467, int_538468)
        
        # Applying the binary operator '+' (line 129)
        result_add_538470 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 15), '+', result_mul_538466, result_pow_538469)
        
        # Getting the type of 'h2' (line 129)
        h2_538471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 'h2', False)
        # Getting the type of 'k2' (line 129)
        k2_538472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'k2', False)
        # Applying the binary operator '*' (line 129)
        result_mul_538473 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 33), '*', h2_538471, k2_538472)
        
        # Applying the binary operator '-' (line 129)
        result_sub_538474 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 31), '-', result_add_538470, result_mul_538473)
        
        # Processing the call keyword arguments (line 129)
        kwargs_538475 = {}
        # Getting the type of 'sqrt' (line 129)
        sqrt_538461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 129)
        sqrt_call_result_538476 = invoke(stypy.reporting.localization.Localization(__file__, 129, 10), sqrt_538461, *[result_sub_538474], **kwargs_538475)
        
        int_538477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 41), 'int')
        # Getting the type of 'h2' (line 129)
        h2_538478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 44), 'h2')
        int_538479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 48), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_538480 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 44), '**', h2_538478, int_538479)
        
        # Applying the binary operator '*' (line 129)
        result_mul_538481 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 41), '*', int_538477, result_pow_538480)
        
        int_538482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 52), 'int')
        # Getting the type of 'k2' (line 129)
        k2_538483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 54), 'k2')
        int_538484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 58), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_538485 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 54), '**', k2_538483, int_538484)
        
        # Applying the binary operator '*' (line 129)
        result_mul_538486 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 52), '*', int_538482, result_pow_538485)
        
        # Applying the binary operator '-' (line 129)
        result_sub_538487 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 41), '-', result_mul_538481, result_mul_538486)
        
        int_538488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 62), 'int')
        # Getting the type of 'h2' (line 129)
        h2_538489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 65), 'h2')
        int_538490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 69), 'int')
        # Applying the binary operator '**' (line 129)
        result_pow_538491 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 65), '**', h2_538489, int_538490)
        
        # Applying the binary operator '*' (line 129)
        result_mul_538492 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 62), '*', int_538488, result_pow_538491)
        
        # Getting the type of 'k2' (line 129)
        k2_538493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 71), 'k2')
        # Applying the binary operator '*' (line 129)
        result_mul_538494 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 70), '*', result_mul_538492, k2_538493)
        
        # Applying the binary operator '+' (line 129)
        result_add_538495 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 60), '+', result_sub_538487, result_mul_538494)
        
        int_538496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        # Getting the type of 'h2' (line 130)
        h2_538497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 10), 'h2')
        # Applying the binary operator '*' (line 130)
        result_mul_538498 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 8), '*', int_538496, h2_538497)
        
        # Getting the type of 'k2' (line 130)
        k2_538499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'k2')
        int_538500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 17), 'int')
        # Applying the binary operator '**' (line 130)
        result_pow_538501 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 13), '**', k2_538499, int_538500)
        
        # Applying the binary operator '*' (line 130)
        result_mul_538502 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 12), '*', result_mul_538498, result_pow_538501)
        
        # Applying the binary operator '+' (line 129)
        result_add_538503 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 74), '+', result_add_538495, result_mul_538502)
        
        # Applying the binary operator '*' (line 129)
        result_mul_538504 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 10), '*', sqrt_call_result_538476, result_add_538503)
        
        # Applying the binary operator '+' (line 129)
        result_add_538505 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 8), '+', result_add_538460, result_mul_538504)
        
        # Assigning a type to the variable 'res' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'res', result_add_538505)
        int_538506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'int')
        # Getting the type of 'pi' (line 131)
        pi_538507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), 'pi')
        # Applying the binary operator '*' (line 131)
        result_mul_538508 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), '*', int_538506, pi_538507)
        
        int_538509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 21), 'int')
        # Applying the binary operator 'div' (line 131)
        result_div_538510 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 20), 'div', result_mul_538508, int_538509)
        
        # Getting the type of 'k2' (line 131)
        k2_538511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'k2')
        # Applying the binary operator '*' (line 131)
        result_mul_538512 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 26), '*', result_div_538510, k2_538511)
        
        # Getting the type of 'k2' (line 131)
        k2_538513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'k2')
        # Getting the type of 'h2' (line 131)
        h2_538514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'h2')
        # Applying the binary operator '-' (line 131)
        result_sub_538515 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 31), '-', k2_538513, h2_538514)
        
        # Applying the binary operator '*' (line 131)
        result_mul_538516 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '*', result_mul_538512, result_sub_538515)
        
        # Getting the type of 'res' (line 131)
        res_538517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'res')
        # Applying the binary operator '*' (line 131)
        result_mul_538518 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 39), '*', result_mul_538516, res_538517)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', result_mul_538518)
        
        # ################# End of 'G36(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G36' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_538519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538519)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G36'
        return stypy_return_type_538519

    # Assigning a type to the variable 'G36' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'G36', G36)

    @norecursion
    def G35(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G35'
        module_type_store = module_type_store.open_function_context('G35', 133, 4, False)
        
        # Passed parameters checking function
        G35.stypy_localization = localization
        G35.stypy_type_of_self = None
        G35.stypy_type_store = module_type_store
        G35.stypy_function_name = 'G35'
        G35.stypy_param_names_list = ['h2', 'k2']
        G35.stypy_varargs_param_name = None
        G35.stypy_kwargs_param_name = None
        G35.stypy_call_defaults = defaults
        G35.stypy_call_varargs = varargs
        G35.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G35', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G35', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G35(...)' code ##################

        
        # Assigning a BinOp to a Name (line 134):
        
        # Assigning a BinOp to a Name (line 134):
        int_538520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'int')
        # Getting the type of 'h2' (line 134)
        h2_538521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'h2')
        int_538522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 22), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538523 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 18), '**', h2_538521, int_538522)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538524 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '*', int_538520, result_pow_538523)
        
        int_538525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'int')
        # Getting the type of 'k2' (line 134)
        k2_538526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'k2')
        int_538527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538528 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 28), '**', k2_538526, int_538527)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538529 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 26), '*', int_538525, result_pow_538528)
        
        # Applying the binary operator '+' (line 134)
        result_add_538530 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '+', result_mul_538524, result_mul_538529)
        
        int_538531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'int')
        # Getting the type of 'h2' (line 134)
        h2_538532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 39), 'h2')
        int_538533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 43), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538534 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 39), '**', h2_538532, int_538533)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538535 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 36), '*', int_538531, result_pow_538534)
        
        # Getting the type of 'k2' (line 134)
        k2_538536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 45), 'k2')
        # Applying the binary operator '*' (line 134)
        result_mul_538537 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 44), '*', result_mul_538535, k2_538536)
        
        # Applying the binary operator '-' (line 134)
        result_sub_538538 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 34), '-', result_add_538530, result_mul_538537)
        
        int_538539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 50), 'int')
        # Getting the type of 'h2' (line 134)
        h2_538540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 53), 'h2')
        # Applying the binary operator '*' (line 134)
        result_mul_538541 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 50), '*', int_538539, h2_538540)
        
        # Getting the type of 'k2' (line 134)
        k2_538542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 56), 'k2')
        int_538543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 60), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538544 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 56), '**', k2_538542, int_538543)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538545 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 55), '*', result_mul_538541, result_pow_538544)
        
        # Applying the binary operator '-' (line 134)
        result_sub_538546 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 48), '-', result_sub_538538, result_mul_538545)
        
        int_538547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 64), 'int')
        # Getting the type of 'h2' (line 134)
        h2_538548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 67), 'h2')
        int_538549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 71), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538550 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 67), '**', h2_538548, int_538549)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538551 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 64), '*', int_538547, result_pow_538550)
        
        # Getting the type of 'k2' (line 134)
        k2_538552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 73), 'k2')
        int_538553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 77), 'int')
        # Applying the binary operator '**' (line 134)
        result_pow_538554 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 73), '**', k2_538552, int_538553)
        
        # Applying the binary operator '*' (line 134)
        result_mul_538555 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 72), '*', result_mul_538551, result_pow_538554)
        
        # Applying the binary operator '+' (line 134)
        result_add_538556 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 62), '+', result_sub_538546, result_mul_538555)
        
        
        # Call to sqrt(...): (line 135)
        # Processing the call arguments (line 135)
        int_538558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'int')
        # Getting the type of 'h2' (line 135)
        h2_538559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'h2', False)
        int_538560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_538561 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 17), '**', h2_538559, int_538560)
        
        # Applying the binary operator '*' (line 135)
        result_mul_538562 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '*', int_538558, result_pow_538561)
        
        # Getting the type of 'k2' (line 135)
        k2_538563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 25), 'k2', False)
        int_538564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_538565 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 25), '**', k2_538563, int_538564)
        
        # Applying the binary operator '+' (line 135)
        result_add_538566 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '+', result_mul_538562, result_pow_538565)
        
        # Getting the type of 'h2' (line 135)
        h2_538567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'h2', False)
        # Getting the type of 'k2' (line 135)
        k2_538568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'k2', False)
        # Applying the binary operator '*' (line 135)
        result_mul_538569 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 33), '*', h2_538567, k2_538568)
        
        # Applying the binary operator '-' (line 135)
        result_sub_538570 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 31), '-', result_add_538566, result_mul_538569)
        
        # Processing the call keyword arguments (line 135)
        kwargs_538571 = {}
        # Getting the type of 'sqrt' (line 135)
        sqrt_538557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 10), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 135)
        sqrt_call_result_538572 = invoke(stypy.reporting.localization.Localization(__file__, 135, 10), sqrt_538557, *[result_sub_538570], **kwargs_538571)
        
        int_538573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 41), 'int')
        # Getting the type of 'h2' (line 135)
        h2_538574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 43), 'h2')
        int_538575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_538576 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 43), '**', h2_538574, int_538575)
        
        # Applying the binary operator '*' (line 135)
        result_mul_538577 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), '*', int_538573, result_pow_538576)
        
        int_538578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 51), 'int')
        # Getting the type of 'k2' (line 135)
        k2_538579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 53), 'k2')
        int_538580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 57), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_538581 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 53), '**', k2_538579, int_538580)
        
        # Applying the binary operator '*' (line 135)
        result_mul_538582 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 51), '*', int_538578, result_pow_538581)
        
        # Applying the binary operator '+' (line 135)
        result_add_538583 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 41), '+', result_mul_538577, result_mul_538582)
        
        int_538584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 61), 'int')
        # Getting the type of 'h2' (line 135)
        h2_538585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 64), 'h2')
        int_538586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 68), 'int')
        # Applying the binary operator '**' (line 135)
        result_pow_538587 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 64), '**', h2_538585, int_538586)
        
        # Applying the binary operator '*' (line 135)
        result_mul_538588 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 61), '*', int_538584, result_pow_538587)
        
        # Getting the type of 'k2' (line 135)
        k2_538589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 70), 'k2')
        # Applying the binary operator '*' (line 135)
        result_mul_538590 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 69), '*', result_mul_538588, k2_538589)
        
        # Applying the binary operator '-' (line 135)
        result_sub_538591 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 59), '-', result_add_538583, result_mul_538590)
        
        int_538592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
        # Getting the type of 'h2' (line 136)
        h2_538593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 10), 'h2')
        # Applying the binary operator '*' (line 136)
        result_mul_538594 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 8), '*', int_538592, h2_538593)
        
        # Getting the type of 'k2' (line 136)
        k2_538595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'k2')
        int_538596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 17), 'int')
        # Applying the binary operator '**' (line 136)
        result_pow_538597 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 13), '**', k2_538595, int_538596)
        
        # Applying the binary operator '*' (line 136)
        result_mul_538598 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 12), '*', result_mul_538594, result_pow_538597)
        
        # Applying the binary operator '-' (line 135)
        result_sub_538599 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 73), '-', result_sub_538591, result_mul_538598)
        
        # Applying the binary operator '*' (line 135)
        result_mul_538600 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 10), '*', sqrt_call_result_538572, result_sub_538599)
        
        # Applying the binary operator '+' (line 135)
        result_add_538601 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 8), '+', result_add_538556, result_mul_538600)
        
        # Assigning a type to the variable 'res' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'res', result_add_538601)
        int_538602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 15), 'int')
        # Getting the type of 'pi' (line 137)
        pi_538603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 'pi')
        # Applying the binary operator '*' (line 137)
        result_mul_538604 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '*', int_538602, pi_538603)
        
        int_538605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 21), 'int')
        # Applying the binary operator 'div' (line 137)
        result_div_538606 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 20), 'div', result_mul_538604, int_538605)
        
        # Getting the type of 'k2' (line 137)
        k2_538607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'k2')
        # Applying the binary operator '*' (line 137)
        result_mul_538608 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 26), '*', result_div_538606, k2_538607)
        
        # Getting the type of 'k2' (line 137)
        k2_538609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'k2')
        # Getting the type of 'h2' (line 137)
        h2_538610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 36), 'h2')
        # Applying the binary operator '-' (line 137)
        result_sub_538611 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 31), '-', k2_538609, h2_538610)
        
        # Applying the binary operator '*' (line 137)
        result_mul_538612 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 29), '*', result_mul_538608, result_sub_538611)
        
        # Getting the type of 'res' (line 137)
        res_538613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'res')
        # Applying the binary operator '*' (line 137)
        result_mul_538614 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 39), '*', result_mul_538612, res_538613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', result_mul_538614)
        
        # ################# End of 'G35(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G35' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_538615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G35'
        return stypy_return_type_538615

    # Assigning a type to the variable 'G35' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'G35', G35)

    @norecursion
    def G37(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'G37'
        module_type_store = module_type_store.open_function_context('G37', 139, 4, False)
        
        # Passed parameters checking function
        G37.stypy_localization = localization
        G37.stypy_type_of_self = None
        G37.stypy_type_store = module_type_store
        G37.stypy_function_name = 'G37'
        G37.stypy_param_names_list = ['h2', 'k2']
        G37.stypy_varargs_param_name = None
        G37.stypy_kwargs_param_name = None
        G37.stypy_call_defaults = defaults
        G37.stypy_call_varargs = varargs
        G37.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'G37', ['h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'G37', localization, ['h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'G37(...)' code ##################

        int_538616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 15), 'int')
        # Getting the type of 'pi' (line 140)
        pi_538617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 17), 'pi')
        # Applying the binary operator '*' (line 140)
        result_mul_538618 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), '*', int_538616, pi_538617)
        
        # Getting the type of 'h2' (line 140)
        h2_538619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'h2')
        int_538620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'int')
        # Applying the binary operator '**' (line 140)
        result_pow_538621 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 20), '**', h2_538619, int_538620)
        
        # Applying the binary operator '*' (line 140)
        result_mul_538622 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 19), '*', result_mul_538618, result_pow_538621)
        
        # Getting the type of 'k2' (line 140)
        k2_538623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 26), 'k2')
        int_538624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 30), 'int')
        # Applying the binary operator '**' (line 140)
        result_pow_538625 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 26), '**', k2_538623, int_538624)
        
        # Applying the binary operator '*' (line 140)
        result_mul_538626 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 25), '*', result_mul_538622, result_pow_538625)
        
        # Getting the type of 'k2' (line 140)
        k2_538627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 33), 'k2')
        # Getting the type of 'h2' (line 140)
        h2_538628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'h2')
        # Applying the binary operator '-' (line 140)
        result_sub_538629 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 33), '-', k2_538627, h2_538628)
        
        int_538630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 43), 'int')
        # Applying the binary operator '**' (line 140)
        result_pow_538631 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 32), '**', result_sub_538629, int_538630)
        
        # Applying the binary operator '*' (line 140)
        result_mul_538632 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 31), '*', result_mul_538626, result_pow_538631)
        
        int_538633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 45), 'int')
        # Applying the binary operator 'div' (line 140)
        result_div_538634 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 44), 'div', result_mul_538632, int_538633)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', result_div_538634)
        
        # ################# End of 'G37(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'G37' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_538635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538635)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'G37'
        return stypy_return_type_538635

    # Assigning a type to the variable 'G37' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'G37', G37)
    
    # Assigning a Dict to a Name (line 142):
    
    # Assigning a Dict to a Name (line 142):
    
    # Obtaining an instance of the builtin type 'dict' (line 142)
    dict_538636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 142)
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_538637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    int_538638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 20), tuple_538637, int_538638)
    # Adding element type (line 142)
    int_538639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 20), tuple_538637, int_538639)
    
    # Getting the type of 'G01' (line 142)
    G01_538640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'G01')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538637, G01_538640))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_538641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    int_538642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), tuple_538641, int_538642)
    # Adding element type (line 142)
    int_538643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), tuple_538641, int_538643)
    
    # Getting the type of 'G11' (line 142)
    G11_538644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 40), 'G11')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538641, G11_538644))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_538645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    int_538646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 46), tuple_538645, int_538646)
    # Adding element type (line 142)
    int_538647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 46), tuple_538645, int_538647)
    
    # Getting the type of 'G12' (line 142)
    G12_538648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 53), 'G12')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538645, G12_538648))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 142)
    tuple_538649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 142)
    # Adding element type (line 142)
    int_538650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 59), tuple_538649, int_538650)
    # Adding element type (line 142)
    int_538651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 59), tuple_538649, int_538651)
    
    # Getting the type of 'G13' (line 142)
    G13_538652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 66), 'G13')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538649, G13_538652))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_538653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_538654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 20), tuple_538653, int_538654)
    # Adding element type (line 143)
    int_538655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 20), tuple_538653, int_538655)
    
    # Getting the type of 'G21' (line 143)
    G21_538656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'G21')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538653, G21_538656))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_538657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_538658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 33), tuple_538657, int_538658)
    # Adding element type (line 143)
    int_538659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 33), tuple_538657, int_538659)
    
    # Getting the type of 'G22' (line 143)
    G22_538660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'G22')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538657, G22_538660))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_538661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_538662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 46), tuple_538661, int_538662)
    # Adding element type (line 143)
    int_538663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 46), tuple_538661, int_538663)
    
    # Getting the type of 'G23' (line 143)
    G23_538664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 53), 'G23')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538661, G23_538664))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_538665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    int_538666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 59), tuple_538665, int_538666)
    # Adding element type (line 143)
    int_538667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 59), tuple_538665, int_538667)
    
    # Getting the type of 'G24' (line 143)
    G24_538668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 66), 'G24')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538665, G24_538668))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_538669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    int_538670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 20), tuple_538669, int_538670)
    # Adding element type (line 144)
    int_538671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 20), tuple_538669, int_538671)
    
    # Getting the type of 'G25' (line 144)
    G25_538672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'G25')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538669, G25_538672))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_538673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    int_538674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 33), tuple_538673, int_538674)
    # Adding element type (line 144)
    int_538675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 33), tuple_538673, int_538675)
    
    # Getting the type of 'G31' (line 144)
    G31_538676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 40), 'G31')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538673, G31_538676))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_538677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    int_538678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 46), tuple_538677, int_538678)
    # Adding element type (line 144)
    int_538679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 46), tuple_538677, int_538679)
    
    # Getting the type of 'G32' (line 144)
    G32_538680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 53), 'G32')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538677, G32_538680))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_538681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    int_538682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 59), tuple_538681, int_538682)
    # Adding element type (line 144)
    int_538683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 59), tuple_538681, int_538683)
    
    # Getting the type of 'G33' (line 144)
    G33_538684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 66), 'G33')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538681, G33_538684))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_538685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    int_538686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 20), tuple_538685, int_538686)
    # Adding element type (line 145)
    int_538687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 20), tuple_538685, int_538687)
    
    # Getting the type of 'G34' (line 145)
    G34_538688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'G34')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538685, G34_538688))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_538689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    int_538690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 33), tuple_538689, int_538690)
    # Adding element type (line 145)
    int_538691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 33), tuple_538689, int_538691)
    
    # Getting the type of 'G35' (line 145)
    G35_538692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 40), 'G35')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538689, G35_538692))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_538693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    int_538694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 46), tuple_538693, int_538694)
    # Adding element type (line 145)
    int_538695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 46), tuple_538693, int_538695)
    
    # Getting the type of 'G36' (line 145)
    G36_538696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 53), 'G36')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538693, G36_538696))
    # Adding element type (key, value) (line 142)
    
    # Obtaining an instance of the builtin type 'tuple' (line 145)
    tuple_538697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 145)
    # Adding element type (line 145)
    int_538698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 59), tuple_538697, int_538698)
    # Adding element type (line 145)
    int_538699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 59), tuple_538697, int_538699)
    
    # Getting the type of 'G37' (line 145)
    G37_538700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 66), 'G37')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 18), dict_538636, (tuple_538697, G37_538700))
    
    # Assigning a type to the variable 'known_funcs' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'known_funcs', dict_538636)

    @norecursion
    def _ellip_norm(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_ellip_norm'
        module_type_store = module_type_store.open_function_context('_ellip_norm', 147, 4, False)
        
        # Passed parameters checking function
        _ellip_norm.stypy_localization = localization
        _ellip_norm.stypy_type_of_self = None
        _ellip_norm.stypy_type_store = module_type_store
        _ellip_norm.stypy_function_name = '_ellip_norm'
        _ellip_norm.stypy_param_names_list = ['n', 'p', 'h2', 'k2']
        _ellip_norm.stypy_varargs_param_name = None
        _ellip_norm.stypy_kwargs_param_name = None
        _ellip_norm.stypy_call_defaults = defaults
        _ellip_norm.stypy_call_varargs = varargs
        _ellip_norm.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_ellip_norm', ['n', 'p', 'h2', 'k2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_ellip_norm', localization, ['n', 'p', 'h2', 'k2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_ellip_norm(...)' code ##################

        
        # Assigning a Subscript to a Name (line 148):
        
        # Assigning a Subscript to a Name (line 148):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_538701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'n' (line 148)
        n_538702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 27), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 27), tuple_538701, n_538702)
        # Adding element type (line 148)
        # Getting the type of 'p' (line 148)
        p_538703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 27), tuple_538701, p_538703)
        
        # Getting the type of 'known_funcs' (line 148)
        known_funcs_538704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'known_funcs')
        # Obtaining the member '__getitem__' of a type (line 148)
        getitem___538705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), known_funcs_538704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 148)
        subscript_call_result_538706 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), getitem___538705, tuple_538701)
        
        # Assigning a type to the variable 'func' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'func', subscript_call_result_538706)
        
        # Call to func(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'h2' (line 149)
        h2_538708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'h2', False)
        # Getting the type of 'k2' (line 149)
        k2_538709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'k2', False)
        # Processing the call keyword arguments (line 149)
        kwargs_538710 = {}
        # Getting the type of 'func' (line 149)
        func_538707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'func', False)
        # Calling func(args, kwargs) (line 149)
        func_call_result_538711 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), func_538707, *[h2_538708, k2_538709], **kwargs_538710)
        
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', func_call_result_538711)
        
        # ################# End of '_ellip_norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_ellip_norm' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_538712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538712)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_ellip_norm'
        return stypy_return_type_538712

    # Assigning a type to the variable '_ellip_norm' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), '_ellip_norm', _ellip_norm)
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 150):
    
    # Call to vectorize(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of '_ellip_norm' (line 150)
    _ellip_norm_538715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), '_ellip_norm', False)
    # Processing the call keyword arguments (line 150)
    kwargs_538716 = {}
    # Getting the type of 'np' (line 150)
    np_538713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'np', False)
    # Obtaining the member 'vectorize' of a type (line 150)
    vectorize_538714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 18), np_538713, 'vectorize')
    # Calling vectorize(args, kwargs) (line 150)
    vectorize_call_result_538717 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), vectorize_538714, *[_ellip_norm_538715], **kwargs_538716)
    
    # Assigning a type to the variable '_ellip_norm' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), '_ellip_norm', vectorize_call_result_538717)

    @norecursion
    def ellip_normal_known(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ellip_normal_known'
        module_type_store = module_type_store.open_function_context('ellip_normal_known', 152, 4, False)
        
        # Passed parameters checking function
        ellip_normal_known.stypy_localization = localization
        ellip_normal_known.stypy_type_of_self = None
        ellip_normal_known.stypy_type_store = module_type_store
        ellip_normal_known.stypy_function_name = 'ellip_normal_known'
        ellip_normal_known.stypy_param_names_list = ['h2', 'k2', 'n', 'p']
        ellip_normal_known.stypy_varargs_param_name = None
        ellip_normal_known.stypy_kwargs_param_name = None
        ellip_normal_known.stypy_call_defaults = defaults
        ellip_normal_known.stypy_call_varargs = varargs
        ellip_normal_known.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'ellip_normal_known', ['h2', 'k2', 'n', 'p'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ellip_normal_known', localization, ['h2', 'k2', 'n', 'p'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ellip_normal_known(...)' code ##################

        
        # Call to _ellip_norm(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'n' (line 153)
        n_538719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'n', False)
        # Getting the type of 'p' (line 153)
        p_538720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'p', False)
        # Getting the type of 'h2' (line 153)
        h2_538721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'h2', False)
        # Getting the type of 'k2' (line 153)
        k2_538722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'k2', False)
        # Processing the call keyword arguments (line 153)
        kwargs_538723 = {}
        # Getting the type of '_ellip_norm' (line 153)
        _ellip_norm_538718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), '_ellip_norm', False)
        # Calling _ellip_norm(args, kwargs) (line 153)
        _ellip_norm_call_result_538724 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), _ellip_norm_538718, *[n_538719, p_538720, h2_538721, k2_538722], **kwargs_538723)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', _ellip_norm_call_result_538724)
        
        # ################# End of 'ellip_normal_known(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ellip_normal_known' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_538725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ellip_normal_known'
        return stypy_return_type_538725

    # Assigning a type to the variable 'ellip_normal_known' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'ellip_normal_known', ellip_normal_known)
    
    # Call to seed(...): (line 156)
    # Processing the call arguments (line 156)
    int_538729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 19), 'int')
    # Processing the call keyword arguments (line 156)
    kwargs_538730 = {}
    # Getting the type of 'np' (line 156)
    np_538726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 156)
    random_538727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 4), np_538726, 'random')
    # Obtaining the member 'seed' of a type (line 156)
    seed_538728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 4), random_538727, 'seed')
    # Calling seed(args, kwargs) (line 156)
    seed_call_result_538731 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), seed_538728, *[int_538729], **kwargs_538730)
    
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to pareto(...): (line 157)
    # Processing the call arguments (line 157)
    float_538735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 26), 'float')
    # Processing the call keyword arguments (line 157)
    int_538736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 36), 'int')
    keyword_538737 = int_538736
    kwargs_538738 = {'size': keyword_538737}
    # Getting the type of 'np' (line 157)
    np_538732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 157)
    random_538733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 9), np_538732, 'random')
    # Obtaining the member 'pareto' of a type (line 157)
    pareto_538734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 9), random_538733, 'pareto')
    # Calling pareto(args, kwargs) (line 157)
    pareto_call_result_538739 = invoke(stypy.reporting.localization.Localization(__file__, 157, 9), pareto_538734, *[float_538735], **kwargs_538738)
    
    # Assigning a type to the variable 'h2' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'h2', pareto_call_result_538739)
    
    # Assigning a BinOp to a Name (line 158):
    
    # Assigning a BinOp to a Name (line 158):
    # Getting the type of 'h2' (line 158)
    h2_538740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 9), 'h2')
    int_538741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'int')
    
    # Call to pareto(...): (line 158)
    # Processing the call arguments (line 158)
    float_538745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'float')
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'h2' (line 158)
    h2_538746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'h2', False)
    # Obtaining the member 'size' of a type (line 158)
    size_538747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 46), h2_538746, 'size')
    keyword_538748 = size_538747
    kwargs_538749 = {'size': keyword_538748}
    # Getting the type of 'np' (line 158)
    np_538742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'np', False)
    # Obtaining the member 'random' of a type (line 158)
    random_538743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 19), np_538742, 'random')
    # Obtaining the member 'pareto' of a type (line 158)
    pareto_538744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 19), random_538743, 'pareto')
    # Calling pareto(args, kwargs) (line 158)
    pareto_call_result_538750 = invoke(stypy.reporting.localization.Localization(__file__, 158, 19), pareto_538744, *[float_538745], **kwargs_538749)
    
    # Applying the binary operator '+' (line 158)
    result_add_538751 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 15), '+', int_538741, pareto_call_result_538750)
    
    # Applying the binary operator '*' (line 158)
    result_mul_538752 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 9), '*', h2_538740, result_add_538751)
    
    # Assigning a type to the variable 'k2' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'k2', result_mul_538752)
    
    # Assigning a List to a Name (line 160):
    
    # Assigning a List to a Name (line 160):
    
    # Obtaining an instance of the builtin type 'list' (line 160)
    list_538753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 160)
    
    # Assigning a type to the variable 'points' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'points', list_538753)
    
    
    # Call to range(...): (line 161)
    # Processing the call arguments (line 161)
    int_538755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 19), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_538756 = {}
    # Getting the type of 'range' (line 161)
    range_538754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'range', False)
    # Calling range(args, kwargs) (line 161)
    range_call_result_538757 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), range_538754, *[int_538755], **kwargs_538756)
    
    # Testing the type of a for loop iterable (line 161)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 161, 4), range_call_result_538757)
    # Getting the type of the for loop variable (line 161)
    for_loop_var_538758 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 161, 4), range_call_result_538757)
    # Assigning a type to the variable 'n' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'n', for_loop_var_538758)
    # SSA begins for a for statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 162)
    # Processing the call arguments (line 162)
    int_538760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
    int_538761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 26), 'int')
    # Getting the type of 'n' (line 162)
    n_538762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'n', False)
    # Applying the binary operator '*' (line 162)
    result_mul_538763 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 26), '*', int_538761, n_538762)
    
    int_538764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 30), 'int')
    # Applying the binary operator '+' (line 162)
    result_add_538765 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 26), '+', result_mul_538763, int_538764)
    
    # Processing the call keyword arguments (line 162)
    kwargs_538766 = {}
    # Getting the type of 'range' (line 162)
    range_538759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'range', False)
    # Calling range(args, kwargs) (line 162)
    range_call_result_538767 = invoke(stypy.reporting.localization.Localization(__file__, 162, 17), range_538759, *[int_538760, result_add_538765], **kwargs_538766)
    
    # Testing the type of a for loop iterable (line 162)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), range_call_result_538767)
    # Getting the type of the for loop variable (line 162)
    for_loop_var_538768 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), range_call_result_538767)
    # Assigning a type to the variable 'p' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'p', for_loop_var_538768)
    # SSA begins for a for statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 163)
    # Processing the call arguments (line 163)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_538771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    # Getting the type of 'h2' (line 163)
    h2_538772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'h2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_538771, h2_538772)
    # Adding element type (line 163)
    # Getting the type of 'k2' (line 163)
    k2_538773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 31), 'k2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_538771, k2_538773)
    # Adding element type (line 163)
    # Getting the type of 'n' (line 163)
    n_538774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 35), 'n', False)
    
    # Call to ones(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'h2' (line 163)
    h2_538777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 45), 'h2', False)
    # Obtaining the member 'size' of a type (line 163)
    size_538778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 45), h2_538777, 'size')
    # Processing the call keyword arguments (line 163)
    kwargs_538779 = {}
    # Getting the type of 'np' (line 163)
    np_538775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'np', False)
    # Obtaining the member 'ones' of a type (line 163)
    ones_538776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 37), np_538775, 'ones')
    # Calling ones(args, kwargs) (line 163)
    ones_call_result_538780 = invoke(stypy.reporting.localization.Localization(__file__, 163, 37), ones_538776, *[size_538778], **kwargs_538779)
    
    # Applying the binary operator '*' (line 163)
    result_mul_538781 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 35), '*', n_538774, ones_call_result_538780)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_538771, result_mul_538781)
    # Adding element type (line 163)
    # Getting the type of 'p' (line 163)
    p_538782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 55), 'p', False)
    
    # Call to ones(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'h2' (line 163)
    h2_538785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 65), 'h2', False)
    # Obtaining the member 'size' of a type (line 163)
    size_538786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 65), h2_538785, 'size')
    # Processing the call keyword arguments (line 163)
    kwargs_538787 = {}
    # Getting the type of 'np' (line 163)
    np_538783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 57), 'np', False)
    # Obtaining the member 'ones' of a type (line 163)
    ones_538784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 57), np_538783, 'ones')
    # Calling ones(args, kwargs) (line 163)
    ones_call_result_538788 = invoke(stypy.reporting.localization.Localization(__file__, 163, 57), ones_538784, *[size_538786], **kwargs_538787)
    
    # Applying the binary operator '*' (line 163)
    result_mul_538789 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 55), '*', p_538782, ones_call_result_538788)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_538771, result_mul_538789)
    
    # Processing the call keyword arguments (line 163)
    kwargs_538790 = {}
    # Getting the type of 'points' (line 163)
    points_538769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'points', False)
    # Obtaining the member 'append' of a type (line 163)
    append_538770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), points_538769, 'append')
    # Calling append(args, kwargs) (line 163)
    append_call_result_538791 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), append_538770, *[tuple_538771], **kwargs_538790)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to array(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'points' (line 164)
    points_538794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'points', False)
    # Processing the call keyword arguments (line 164)
    kwargs_538795 = {}
    # Getting the type of 'np' (line 164)
    np_538792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 164)
    array_538793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), np_538792, 'array')
    # Calling array(args, kwargs) (line 164)
    array_call_result_538796 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), array_538793, *[points_538794], **kwargs_538795)
    
    # Assigning a type to the variable 'points' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'points', array_call_result_538796)
    
    # Call to suppress_warnings(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_538798 = {}
    # Getting the type of 'suppress_warnings' (line 165)
    suppress_warnings_538797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 165)
    suppress_warnings_call_result_538799 = invoke(stypy.reporting.localization.Localization(__file__, 165, 9), suppress_warnings_538797, *[], **kwargs_538798)
    
    with_538800 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 165, 9), suppress_warnings_call_result_538799, 'with parameter', '__enter__', '__exit__')

    if with_538800:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 165)
        enter___538801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 9), suppress_warnings_call_result_538799, '__enter__')
        with_enter_538802 = invoke(stypy.reporting.localization.Localization(__file__, 165, 9), enter___538801)
        # Assigning a type to the variable 'sup' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 9), 'sup', with_enter_538802)
        
        # Call to filter(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'IntegrationWarning' (line 166)
        IntegrationWarning_538805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 'IntegrationWarning', False)
        str_538806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 39), 'str', 'The occurrence of roundoff error')
        # Processing the call keyword arguments (line 166)
        kwargs_538807 = {}
        # Getting the type of 'sup' (line 166)
        sup_538803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 166)
        filter_538804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), sup_538803, 'filter')
        # Calling filter(args, kwargs) (line 166)
        filter_call_result_538808 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), filter_538804, *[IntegrationWarning_538805, str_538806], **kwargs_538807)
        
        
        # Call to assert_func_equal(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'ellip_normal' (line 167)
        ellip_normal_538810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 26), 'ellip_normal', False)
        # Getting the type of 'ellip_normal_known' (line 167)
        ellip_normal_known_538811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 40), 'ellip_normal_known', False)
        # Getting the type of 'points' (line 167)
        points_538812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 60), 'points', False)
        # Processing the call keyword arguments (line 167)
        float_538813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 73), 'float')
        keyword_538814 = float_538813
        kwargs_538815 = {'rtol': keyword_538814}
        # Getting the type of 'assert_func_equal' (line 167)
        assert_func_equal_538809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_func_equal', False)
        # Calling assert_func_equal(args, kwargs) (line 167)
        assert_func_equal_call_result_538816 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_func_equal_538809, *[ellip_normal_538810, ellip_normal_known_538811, points_538812], **kwargs_538815)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 165)
        exit___538817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 9), suppress_warnings_call_result_538799, '__exit__')
        with_exit_538818 = invoke(stypy.reporting.localization.Localization(__file__, 165, 9), exit___538817, None, None, None)

    
    # ################# End of 'test_ellip_norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ellip_norm' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_538819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_538819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ellip_norm'
    return stypy_return_type_538819

# Assigning a type to the variable 'test_ellip_norm' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'test_ellip_norm', test_ellip_norm)

@norecursion
def test_ellip_harm_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ellip_harm_2'
    module_type_store = module_type_store.open_function_context('test_ellip_harm_2', 170, 0, False)
    
    # Passed parameters checking function
    test_ellip_harm_2.stypy_localization = localization
    test_ellip_harm_2.stypy_type_of_self = None
    test_ellip_harm_2.stypy_type_store = module_type_store
    test_ellip_harm_2.stypy_function_name = 'test_ellip_harm_2'
    test_ellip_harm_2.stypy_param_names_list = []
    test_ellip_harm_2.stypy_varargs_param_name = None
    test_ellip_harm_2.stypy_kwargs_param_name = None
    test_ellip_harm_2.stypy_call_defaults = defaults
    test_ellip_harm_2.stypy_call_varargs = varargs
    test_ellip_harm_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ellip_harm_2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ellip_harm_2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ellip_harm_2(...)' code ##################


    @norecursion
    def I1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'I1'
        module_type_store = module_type_store.open_function_context('I1', 172, 4, False)
        
        # Passed parameters checking function
        I1.stypy_localization = localization
        I1.stypy_type_of_self = None
        I1.stypy_type_store = module_type_store
        I1.stypy_function_name = 'I1'
        I1.stypy_param_names_list = ['h2', 'k2', 's']
        I1.stypy_varargs_param_name = None
        I1.stypy_kwargs_param_name = None
        I1.stypy_call_defaults = defaults
        I1.stypy_call_varargs = varargs
        I1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'I1', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'I1', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'I1(...)' code ##################

        
        # Assigning a BinOp to a Name (line 173):
        
        # Assigning a BinOp to a Name (line 173):
        
        # Call to ellip_harm_2(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'h2' (line 173)
        h2_538821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'h2', False)
        # Getting the type of 'k2' (line 173)
        k2_538822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'k2', False)
        int_538823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 36), 'int')
        int_538824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 39), 'int')
        # Getting the type of 's' (line 173)
        s_538825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 42), 's', False)
        # Processing the call keyword arguments (line 173)
        kwargs_538826 = {}
        # Getting the type of 'ellip_harm_2' (line 173)
        ellip_harm_2_538820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 173)
        ellip_harm_2_call_result_538827 = invoke(stypy.reporting.localization.Localization(__file__, 173, 15), ellip_harm_2_538820, *[h2_538821, k2_538822, int_538823, int_538824, s_538825], **kwargs_538826)
        
        int_538828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 46), 'int')
        
        # Call to ellip_harm(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'h2' (line 173)
        h2_538830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 61), 'h2', False)
        # Getting the type of 'k2' (line 173)
        k2_538831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 65), 'k2', False)
        int_538832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 69), 'int')
        int_538833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 72), 'int')
        # Getting the type of 's' (line 173)
        s_538834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 75), 's', False)
        # Processing the call keyword arguments (line 173)
        kwargs_538835 = {}
        # Getting the type of 'ellip_harm' (line 173)
        ellip_harm_538829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 50), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 173)
        ellip_harm_call_result_538836 = invoke(stypy.reporting.localization.Localization(__file__, 173, 50), ellip_harm_538829, *[h2_538830, k2_538831, int_538832, int_538833, s_538834], **kwargs_538835)
        
        # Applying the binary operator '*' (line 173)
        result_mul_538837 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 46), '*', int_538828, ellip_harm_call_result_538836)
        
        # Applying the binary operator 'div' (line 173)
        result_div_538838 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), 'div', ellip_harm_2_call_result_538827, result_mul_538837)
        
        
        # Call to ellip_harm_2(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'h2' (line 174)
        h2_538840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 23), 'h2', False)
        # Getting the type of 'k2' (line 174)
        k2_538841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'k2', False)
        int_538842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 31), 'int')
        int_538843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'int')
        # Getting the type of 's' (line 174)
        s_538844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 37), 's', False)
        # Processing the call keyword arguments (line 174)
        kwargs_538845 = {}
        # Getting the type of 'ellip_harm_2' (line 174)
        ellip_harm_2_538839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 174)
        ellip_harm_2_call_result_538846 = invoke(stypy.reporting.localization.Localization(__file__, 174, 10), ellip_harm_2_538839, *[h2_538840, k2_538841, int_538842, int_538843, s_538844], **kwargs_538845)
        
        int_538847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 41), 'int')
        
        # Call to ellip_harm(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'h2' (line 174)
        h2_538849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 56), 'h2', False)
        # Getting the type of 'k2' (line 174)
        k2_538850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 60), 'k2', False)
        int_538851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 64), 'int')
        int_538852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 67), 'int')
        # Getting the type of 's' (line 174)
        s_538853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 70), 's', False)
        # Processing the call keyword arguments (line 174)
        kwargs_538854 = {}
        # Getting the type of 'ellip_harm' (line 174)
        ellip_harm_538848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 45), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 174)
        ellip_harm_call_result_538855 = invoke(stypy.reporting.localization.Localization(__file__, 174, 45), ellip_harm_538848, *[h2_538849, k2_538850, int_538851, int_538852, s_538853], **kwargs_538854)
        
        # Applying the binary operator '*' (line 174)
        result_mul_538856 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 41), '*', int_538847, ellip_harm_call_result_538855)
        
        # Applying the binary operator 'div' (line 174)
        result_div_538857 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 10), 'div', ellip_harm_2_call_result_538846, result_mul_538856)
        
        # Applying the binary operator '+' (line 173)
        result_add_538858 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '+', result_div_538838, result_div_538857)
        
        
        # Call to ellip_harm_2(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'h2' (line 175)
        h2_538860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'h2', False)
        # Getting the type of 'k2' (line 175)
        k2_538861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'k2', False)
        int_538862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'int')
        int_538863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'int')
        # Getting the type of 's' (line 175)
        s_538864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 's', False)
        # Processing the call keyword arguments (line 175)
        kwargs_538865 = {}
        # Getting the type of 'ellip_harm_2' (line 175)
        ellip_harm_2_538859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 175)
        ellip_harm_2_call_result_538866 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), ellip_harm_2_538859, *[h2_538860, k2_538861, int_538862, int_538863, s_538864], **kwargs_538865)
        
        int_538867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 39), 'int')
        
        # Call to ellip_harm(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'h2' (line 175)
        h2_538869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 54), 'h2', False)
        # Getting the type of 'k2' (line 175)
        k2_538870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 58), 'k2', False)
        int_538871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 62), 'int')
        int_538872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 65), 'int')
        # Getting the type of 's' (line 175)
        s_538873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 68), 's', False)
        # Processing the call keyword arguments (line 175)
        kwargs_538874 = {}
        # Getting the type of 'ellip_harm' (line 175)
        ellip_harm_538868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'ellip_harm', False)
        # Calling ellip_harm(args, kwargs) (line 175)
        ellip_harm_call_result_538875 = invoke(stypy.reporting.localization.Localization(__file__, 175, 43), ellip_harm_538868, *[h2_538869, k2_538870, int_538871, int_538872, s_538873], **kwargs_538874)
        
        # Applying the binary operator '*' (line 175)
        result_mul_538876 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 39), '*', int_538867, ellip_harm_call_result_538875)
        
        # Applying the binary operator 'div' (line 175)
        result_div_538877 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 8), 'div', ellip_harm_2_call_result_538866, result_mul_538876)
        
        # Applying the binary operator '+' (line 174)
        result_add_538878 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 74), '+', result_add_538858, result_div_538877)
        
        # Assigning a type to the variable 'res' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'res', result_add_538878)
        # Getting the type of 'res' (line 176)
        res_538879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', res_538879)
        
        # ################# End of 'I1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'I1' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_538880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538880)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'I1'
        return stypy_return_type_538880

    # Assigning a type to the variable 'I1' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'I1', I1)
    
    # Call to suppress_warnings(...): (line 178)
    # Processing the call keyword arguments (line 178)
    kwargs_538882 = {}
    # Getting the type of 'suppress_warnings' (line 178)
    suppress_warnings_538881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 178)
    suppress_warnings_call_result_538883 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), suppress_warnings_538881, *[], **kwargs_538882)
    
    with_538884 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 178, 9), suppress_warnings_call_result_538883, 'with parameter', '__enter__', '__exit__')

    if with_538884:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 178)
        enter___538885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 9), suppress_warnings_call_result_538883, '__enter__')
        with_enter_538886 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), enter___538885)
        # Assigning a type to the variable 'sup' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'sup', with_enter_538886)
        
        # Call to filter(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'IntegrationWarning' (line 179)
        IntegrationWarning_538889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'IntegrationWarning', False)
        str_538890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 39), 'str', 'The occurrence of roundoff error')
        # Processing the call keyword arguments (line 179)
        kwargs_538891 = {}
        # Getting the type of 'sup' (line 179)
        sup_538887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 179)
        filter_538888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), sup_538887, 'filter')
        # Calling filter(args, kwargs) (line 179)
        filter_call_result_538892 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), filter_538888, *[IntegrationWarning_538889, str_538890], **kwargs_538891)
        
        
        # Call to assert_almost_equal(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to I1(...): (line 180)
        # Processing the call arguments (line 180)
        int_538895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'int')
        int_538896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 34), 'int')
        int_538897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 37), 'int')
        # Processing the call keyword arguments (line 180)
        kwargs_538898 = {}
        # Getting the type of 'I1' (line 180)
        I1_538894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'I1', False)
        # Calling I1(args, kwargs) (line 180)
        I1_call_result_538899 = invoke(stypy.reporting.localization.Localization(__file__, 180, 28), I1_538894, *[int_538895, int_538896, int_538897], **kwargs_538898)
        
        int_538900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 42), 'int')
        int_538901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 45), 'int')
        
        # Call to sqrt(...): (line 180)
        # Processing the call arguments (line 180)
        int_538903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 54), 'int')
        int_538904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 58), 'int')
        # Applying the binary operator '-' (line 180)
        result_sub_538905 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 54), '-', int_538903, int_538904)
        
        int_538906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 62), 'int')
        int_538907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 66), 'int')
        # Applying the binary operator '-' (line 180)
        result_sub_538908 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 62), '-', int_538906, int_538907)
        
        # Applying the binary operator '*' (line 180)
        result_mul_538909 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 53), '*', result_sub_538905, result_sub_538908)
        
        # Processing the call keyword arguments (line 180)
        kwargs_538910 = {}
        # Getting the type of 'sqrt' (line 180)
        sqrt_538902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 48), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 180)
        sqrt_call_result_538911 = invoke(stypy.reporting.localization.Localization(__file__, 180, 48), sqrt_538902, *[result_mul_538909], **kwargs_538910)
        
        # Applying the binary operator '*' (line 180)
        result_mul_538912 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 45), '*', int_538901, sqrt_call_result_538911)
        
        # Applying the binary operator 'div' (line 180)
        result_div_538913 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 42), 'div', int_538900, result_mul_538912)
        
        # Processing the call keyword arguments (line 180)
        kwargs_538914 = {}
        # Getting the type of 'assert_almost_equal' (line 180)
        assert_almost_equal_538893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 180)
        assert_almost_equal_call_result_538915 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), assert_almost_equal_538893, *[I1_call_result_538899, result_div_538913], **kwargs_538914)
        
        
        # Call to assert_almost_equal(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Call to ellip_harm_2(...): (line 183)
        # Processing the call arguments (line 183)
        int_538918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'int')
        int_538919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 44), 'int')
        int_538920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 47), 'int')
        int_538921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 50), 'int')
        int_538922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 53), 'int')
        # Processing the call keyword arguments (line 183)
        kwargs_538923 = {}
        # Getting the type of 'ellip_harm_2' (line 183)
        ellip_harm_2_538917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 183)
        ellip_harm_2_call_result_538924 = invoke(stypy.reporting.localization.Localization(__file__, 183, 28), ellip_harm_2_538917, *[int_538918, int_538919, int_538920, int_538921, int_538922], **kwargs_538923)
        
        float_538925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 58), 'float')
        # Processing the call keyword arguments (line 183)
        kwargs_538926 = {}
        # Getting the type of 'assert_almost_equal' (line 183)
        assert_almost_equal_538916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 183)
        assert_almost_equal_call_result_538927 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), assert_almost_equal_538916, *[ellip_harm_2_call_result_538924, float_538925], **kwargs_538926)
        
        
        # Call to assert_almost_equal(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to ellip_harm_2(...): (line 184)
        # Processing the call arguments (line 184)
        int_538930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 41), 'int')
        int_538931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'int')
        int_538932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 47), 'int')
        int_538933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 50), 'int')
        int_538934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 53), 'int')
        # Processing the call keyword arguments (line 184)
        kwargs_538935 = {}
        # Getting the type of 'ellip_harm_2' (line 184)
        ellip_harm_2_538929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 184)
        ellip_harm_2_call_result_538936 = invoke(stypy.reporting.localization.Localization(__file__, 184, 28), ellip_harm_2_538929, *[int_538930, int_538931, int_538932, int_538933, int_538934], **kwargs_538935)
        
        float_538937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 58), 'float')
        # Processing the call keyword arguments (line 184)
        kwargs_538938 = {}
        # Getting the type of 'assert_almost_equal' (line 184)
        assert_almost_equal_538928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 184)
        assert_almost_equal_call_result_538939 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert_almost_equal_538928, *[ellip_harm_2_call_result_538936, float_538937], **kwargs_538938)
        
        
        # Call to assert_almost_equal(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Call to ellip_harm_2(...): (line 185)
        # Processing the call arguments (line 185)
        int_538942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 41), 'int')
        int_538943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 44), 'int')
        int_538944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 47), 'int')
        int_538945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 50), 'int')
        int_538946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'int')
        # Processing the call keyword arguments (line 185)
        kwargs_538947 = {}
        # Getting the type of 'ellip_harm_2' (line 185)
        ellip_harm_2_538941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 185)
        ellip_harm_2_call_result_538948 = invoke(stypy.reporting.localization.Localization(__file__, 185, 28), ellip_harm_2_538941, *[int_538942, int_538943, int_538944, int_538945, int_538946], **kwargs_538947)
        
        float_538949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 58), 'float')
        # Processing the call keyword arguments (line 185)
        kwargs_538950 = {}
        # Getting the type of 'assert_almost_equal' (line 185)
        assert_almost_equal_538940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 185)
        assert_almost_equal_call_result_538951 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_almost_equal_538940, *[ellip_harm_2_call_result_538948, float_538949], **kwargs_538950)
        
        
        # Call to assert_almost_equal(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to ellip_harm_2(...): (line 186)
        # Processing the call arguments (line 186)
        int_538954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 41), 'int')
        int_538955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 44), 'int')
        int_538956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 47), 'int')
        int_538957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 50), 'int')
        int_538958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 53), 'int')
        # Processing the call keyword arguments (line 186)
        kwargs_538959 = {}
        # Getting the type of 'ellip_harm_2' (line 186)
        ellip_harm_2_538953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 186)
        ellip_harm_2_call_result_538960 = invoke(stypy.reporting.localization.Localization(__file__, 186, 28), ellip_harm_2_538953, *[int_538954, int_538955, int_538956, int_538957, int_538958], **kwargs_538959)
        
        float_538961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 58), 'float')
        # Processing the call keyword arguments (line 186)
        kwargs_538962 = {}
        # Getting the type of 'assert_almost_equal' (line 186)
        assert_almost_equal_538952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 186)
        assert_almost_equal_call_result_538963 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert_almost_equal_538952, *[ellip_harm_2_call_result_538960, float_538961], **kwargs_538962)
        
        
        # Call to assert_almost_equal(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to ellip_harm_2(...): (line 187)
        # Processing the call arguments (line 187)
        int_538966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 41), 'int')
        int_538967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 44), 'int')
        int_538968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 47), 'int')
        int_538969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 50), 'int')
        int_538970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 53), 'int')
        # Processing the call keyword arguments (line 187)
        kwargs_538971 = {}
        # Getting the type of 'ellip_harm_2' (line 187)
        ellip_harm_2_538965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'ellip_harm_2', False)
        # Calling ellip_harm_2(args, kwargs) (line 187)
        ellip_harm_2_call_result_538972 = invoke(stypy.reporting.localization.Localization(__file__, 187, 28), ellip_harm_2_538965, *[int_538966, int_538967, int_538968, int_538969, int_538970], **kwargs_538971)
        
        float_538973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 58), 'float')
        # Processing the call keyword arguments (line 187)
        kwargs_538974 = {}
        # Getting the type of 'assert_almost_equal' (line 187)
        assert_almost_equal_538964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 187)
        assert_almost_equal_call_result_538975 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_almost_equal_538964, *[ellip_harm_2_call_result_538972, float_538973], **kwargs_538974)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 178)
        exit___538976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 9), suppress_warnings_call_result_538883, '__exit__')
        with_exit_538977 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), exit___538976, None, None, None)

    
    # ################# End of 'test_ellip_harm_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ellip_harm_2' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_538978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_538978)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ellip_harm_2'
    return stypy_return_type_538978

# Assigning a type to the variable 'test_ellip_harm_2' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'test_ellip_harm_2', test_ellip_harm_2)

@norecursion
def test_ellip_harm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ellip_harm'
    module_type_store = module_type_store.open_function_context('test_ellip_harm', 190, 0, False)
    
    # Passed parameters checking function
    test_ellip_harm.stypy_localization = localization
    test_ellip_harm.stypy_type_of_self = None
    test_ellip_harm.stypy_type_store = module_type_store
    test_ellip_harm.stypy_function_name = 'test_ellip_harm'
    test_ellip_harm.stypy_param_names_list = []
    test_ellip_harm.stypy_varargs_param_name = None
    test_ellip_harm.stypy_kwargs_param_name = None
    test_ellip_harm.stypy_call_defaults = defaults
    test_ellip_harm.stypy_call_varargs = varargs
    test_ellip_harm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ellip_harm', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ellip_harm', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ellip_harm(...)' code ##################


    @norecursion
    def E01(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E01'
        module_type_store = module_type_store.open_function_context('E01', 192, 4, False)
        
        # Passed parameters checking function
        E01.stypy_localization = localization
        E01.stypy_type_of_self = None
        E01.stypy_type_store = module_type_store
        E01.stypy_function_name = 'E01'
        E01.stypy_param_names_list = ['h2', 'k2', 's']
        E01.stypy_varargs_param_name = None
        E01.stypy_kwargs_param_name = None
        E01.stypy_call_defaults = defaults
        E01.stypy_call_varargs = varargs
        E01.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E01', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E01', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E01(...)' code ##################

        int_538979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', int_538979)
        
        # ################# End of 'E01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E01' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_538980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E01'
        return stypy_return_type_538980

    # Assigning a type to the variable 'E01' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'E01', E01)

    @norecursion
    def E11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E11'
        module_type_store = module_type_store.open_function_context('E11', 195, 4, False)
        
        # Passed parameters checking function
        E11.stypy_localization = localization
        E11.stypy_type_of_self = None
        E11.stypy_type_store = module_type_store
        E11.stypy_function_name = 'E11'
        E11.stypy_param_names_list = ['h2', 'k2', 's']
        E11.stypy_varargs_param_name = None
        E11.stypy_kwargs_param_name = None
        E11.stypy_call_defaults = defaults
        E11.stypy_call_varargs = varargs
        E11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E11', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E11', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E11(...)' code ##################

        # Getting the type of 's' (line 196)
        s_538981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stypy_return_type', s_538981)
        
        # ################# End of 'E11(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E11' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_538982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538982)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E11'
        return stypy_return_type_538982

    # Assigning a type to the variable 'E11' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'E11', E11)

    @norecursion
    def E12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E12'
        module_type_store = module_type_store.open_function_context('E12', 198, 4, False)
        
        # Passed parameters checking function
        E12.stypy_localization = localization
        E12.stypy_type_of_self = None
        E12.stypy_type_store = module_type_store
        E12.stypy_function_name = 'E12'
        E12.stypy_param_names_list = ['h2', 'k2', 's']
        E12.stypy_varargs_param_name = None
        E12.stypy_kwargs_param_name = None
        E12.stypy_call_defaults = defaults
        E12.stypy_call_varargs = varargs
        E12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E12', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E12', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E12(...)' code ##################

        
        # Call to sqrt(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to abs(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 's' (line 199)
        s_538985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 's', False)
        # Getting the type of 's' (line 199)
        s_538986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 's', False)
        # Applying the binary operator '*' (line 199)
        result_mul_538987 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 24), '*', s_538985, s_538986)
        
        # Getting the type of 'h2' (line 199)
        h2_538988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'h2', False)
        # Applying the binary operator '-' (line 199)
        result_sub_538989 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 24), '-', result_mul_538987, h2_538988)
        
        # Processing the call keyword arguments (line 199)
        kwargs_538990 = {}
        # Getting the type of 'abs' (line 199)
        abs_538984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 199)
        abs_call_result_538991 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), abs_538984, *[result_sub_538989], **kwargs_538990)
        
        # Processing the call keyword arguments (line 199)
        kwargs_538992 = {}
        # Getting the type of 'sqrt' (line 199)
        sqrt_538983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 199)
        sqrt_call_result_538993 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), sqrt_538983, *[abs_call_result_538991], **kwargs_538992)
        
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', sqrt_call_result_538993)
        
        # ################# End of 'E12(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E12' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_538994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_538994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E12'
        return stypy_return_type_538994

    # Assigning a type to the variable 'E12' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'E12', E12)

    @norecursion
    def E13(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E13'
        module_type_store = module_type_store.open_function_context('E13', 201, 4, False)
        
        # Passed parameters checking function
        E13.stypy_localization = localization
        E13.stypy_type_of_self = None
        E13.stypy_type_store = module_type_store
        E13.stypy_function_name = 'E13'
        E13.stypy_param_names_list = ['h2', 'k2', 's']
        E13.stypy_varargs_param_name = None
        E13.stypy_kwargs_param_name = None
        E13.stypy_call_defaults = defaults
        E13.stypy_call_varargs = varargs
        E13.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E13', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E13', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E13(...)' code ##################

        
        # Call to sqrt(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to abs(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 's' (line 202)
        s_538997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 's', False)
        # Getting the type of 's' (line 202)
        s_538998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 26), 's', False)
        # Applying the binary operator '*' (line 202)
        result_mul_538999 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 24), '*', s_538997, s_538998)
        
        # Getting the type of 'k2' (line 202)
        k2_539000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'k2', False)
        # Applying the binary operator '-' (line 202)
        result_sub_539001 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 24), '-', result_mul_538999, k2_539000)
        
        # Processing the call keyword arguments (line 202)
        kwargs_539002 = {}
        # Getting the type of 'abs' (line 202)
        abs_538996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 202)
        abs_call_result_539003 = invoke(stypy.reporting.localization.Localization(__file__, 202, 20), abs_538996, *[result_sub_539001], **kwargs_539002)
        
        # Processing the call keyword arguments (line 202)
        kwargs_539004 = {}
        # Getting the type of 'sqrt' (line 202)
        sqrt_538995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 202)
        sqrt_call_result_539005 = invoke(stypy.reporting.localization.Localization(__file__, 202, 15), sqrt_538995, *[abs_call_result_539003], **kwargs_539004)
        
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', sqrt_call_result_539005)
        
        # ################# End of 'E13(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E13' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_539006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539006)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E13'
        return stypy_return_type_539006

    # Assigning a type to the variable 'E13' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'E13', E13)

    @norecursion
    def E21(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E21'
        module_type_store = module_type_store.open_function_context('E21', 204, 4, False)
        
        # Passed parameters checking function
        E21.stypy_localization = localization
        E21.stypy_type_of_self = None
        E21.stypy_type_store = module_type_store
        E21.stypy_function_name = 'E21'
        E21.stypy_param_names_list = ['h2', 'k2', 's']
        E21.stypy_varargs_param_name = None
        E21.stypy_kwargs_param_name = None
        E21.stypy_call_defaults = defaults
        E21.stypy_call_varargs = varargs
        E21.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E21', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E21', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E21(...)' code ##################

        # Getting the type of 's' (line 205)
        s_539007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 's')
        # Getting the type of 's' (line 205)
        s_539008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 's')
        # Applying the binary operator '*' (line 205)
        result_mul_539009 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 15), '*', s_539007, s_539008)
        
        int_539010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 21), 'int')
        int_539011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'int')
        # Applying the binary operator 'div' (line 205)
        result_div_539012 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 21), 'div', int_539010, int_539011)
        
        # Getting the type of 'h2' (line 205)
        h2_539013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'h2')
        # Getting the type of 'k2' (line 205)
        k2_539014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'k2')
        # Applying the binary operator '+' (line 205)
        result_add_539015 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 27), '+', h2_539013, k2_539014)
        
        
        # Call to sqrt(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to abs(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'h2' (line 205)
        h2_539018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 48), 'h2', False)
        # Getting the type of 'k2' (line 205)
        k2_539019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 53), 'k2', False)
        # Applying the binary operator '+' (line 205)
        result_add_539020 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 48), '+', h2_539018, k2_539019)
        
        # Getting the type of 'h2' (line 205)
        h2_539021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 58), 'h2', False)
        # Getting the type of 'k2' (line 205)
        k2_539022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 63), 'k2', False)
        # Applying the binary operator '+' (line 205)
        result_add_539023 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 58), '+', h2_539021, k2_539022)
        
        # Applying the binary operator '*' (line 205)
        result_mul_539024 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 47), '*', result_add_539020, result_add_539023)
        
        int_539025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 67), 'int')
        # Getting the type of 'h2' (line 205)
        h2_539026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 69), 'h2', False)
        # Applying the binary operator '*' (line 205)
        result_mul_539027 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 67), '*', int_539025, h2_539026)
        
        # Getting the type of 'k2' (line 205)
        k2_539028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 72), 'k2', False)
        # Applying the binary operator '*' (line 205)
        result_mul_539029 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 71), '*', result_mul_539027, k2_539028)
        
        # Applying the binary operator '-' (line 205)
        result_sub_539030 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 47), '-', result_mul_539024, result_mul_539029)
        
        # Processing the call keyword arguments (line 205)
        kwargs_539031 = {}
        # Getting the type of 'abs' (line 205)
        abs_539017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 43), 'abs', False)
        # Calling abs(args, kwargs) (line 205)
        abs_call_result_539032 = invoke(stypy.reporting.localization.Localization(__file__, 205, 43), abs_539017, *[result_sub_539030], **kwargs_539031)
        
        # Processing the call keyword arguments (line 205)
        kwargs_539033 = {}
        # Getting the type of 'sqrt' (line 205)
        sqrt_539016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 38), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 205)
        sqrt_call_result_539034 = invoke(stypy.reporting.localization.Localization(__file__, 205, 38), sqrt_539016, *[abs_call_result_539032], **kwargs_539033)
        
        # Applying the binary operator '+' (line 205)
        result_add_539035 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 26), '+', result_add_539015, sqrt_call_result_539034)
        
        # Applying the binary operator '*' (line 205)
        result_mul_539036 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 24), '*', result_div_539012, result_add_539035)
        
        # Applying the binary operator '-' (line 205)
        result_sub_539037 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 15), '-', result_mul_539009, result_mul_539036)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', result_sub_539037)
        
        # ################# End of 'E21(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E21' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_539038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539038)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E21'
        return stypy_return_type_539038

    # Assigning a type to the variable 'E21' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'E21', E21)

    @norecursion
    def E22(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E22'
        module_type_store = module_type_store.open_function_context('E22', 207, 4, False)
        
        # Passed parameters checking function
        E22.stypy_localization = localization
        E22.stypy_type_of_self = None
        E22.stypy_type_store = module_type_store
        E22.stypy_function_name = 'E22'
        E22.stypy_param_names_list = ['h2', 'k2', 's']
        E22.stypy_varargs_param_name = None
        E22.stypy_kwargs_param_name = None
        E22.stypy_call_defaults = defaults
        E22.stypy_call_varargs = varargs
        E22.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E22', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E22', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E22(...)' code ##################

        # Getting the type of 's' (line 208)
        s_539039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 's')
        # Getting the type of 's' (line 208)
        s_539040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 17), 's')
        # Applying the binary operator '*' (line 208)
        result_mul_539041 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '*', s_539039, s_539040)
        
        int_539042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 21), 'int')
        int_539043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 23), 'int')
        # Applying the binary operator 'div' (line 208)
        result_div_539044 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 21), 'div', int_539042, int_539043)
        
        # Getting the type of 'h2' (line 208)
        h2_539045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 27), 'h2')
        # Getting the type of 'k2' (line 208)
        k2_539046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'k2')
        # Applying the binary operator '+' (line 208)
        result_add_539047 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 27), '+', h2_539045, k2_539046)
        
        
        # Call to sqrt(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Call to abs(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'h2' (line 208)
        h2_539050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 48), 'h2', False)
        # Getting the type of 'k2' (line 208)
        k2_539051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 53), 'k2', False)
        # Applying the binary operator '+' (line 208)
        result_add_539052 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 48), '+', h2_539050, k2_539051)
        
        # Getting the type of 'h2' (line 208)
        h2_539053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 58), 'h2', False)
        # Getting the type of 'k2' (line 208)
        k2_539054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 63), 'k2', False)
        # Applying the binary operator '+' (line 208)
        result_add_539055 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 58), '+', h2_539053, k2_539054)
        
        # Applying the binary operator '*' (line 208)
        result_mul_539056 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 47), '*', result_add_539052, result_add_539055)
        
        int_539057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 67), 'int')
        # Getting the type of 'h2' (line 208)
        h2_539058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 69), 'h2', False)
        # Applying the binary operator '*' (line 208)
        result_mul_539059 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 67), '*', int_539057, h2_539058)
        
        # Getting the type of 'k2' (line 208)
        k2_539060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 72), 'k2', False)
        # Applying the binary operator '*' (line 208)
        result_mul_539061 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 71), '*', result_mul_539059, k2_539060)
        
        # Applying the binary operator '-' (line 208)
        result_sub_539062 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 47), '-', result_mul_539056, result_mul_539061)
        
        # Processing the call keyword arguments (line 208)
        kwargs_539063 = {}
        # Getting the type of 'abs' (line 208)
        abs_539049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 43), 'abs', False)
        # Calling abs(args, kwargs) (line 208)
        abs_call_result_539064 = invoke(stypy.reporting.localization.Localization(__file__, 208, 43), abs_539049, *[result_sub_539062], **kwargs_539063)
        
        # Processing the call keyword arguments (line 208)
        kwargs_539065 = {}
        # Getting the type of 'sqrt' (line 208)
        sqrt_539048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 38), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 208)
        sqrt_call_result_539066 = invoke(stypy.reporting.localization.Localization(__file__, 208, 38), sqrt_539048, *[abs_call_result_539064], **kwargs_539065)
        
        # Applying the binary operator '-' (line 208)
        result_sub_539067 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 26), '-', result_add_539047, sqrt_call_result_539066)
        
        # Applying the binary operator '*' (line 208)
        result_mul_539068 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 24), '*', result_div_539044, result_sub_539067)
        
        # Applying the binary operator '-' (line 208)
        result_sub_539069 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '-', result_mul_539041, result_mul_539068)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', result_sub_539069)
        
        # ################# End of 'E22(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E22' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_539070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E22'
        return stypy_return_type_539070

    # Assigning a type to the variable 'E22' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'E22', E22)

    @norecursion
    def E23(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E23'
        module_type_store = module_type_store.open_function_context('E23', 210, 4, False)
        
        # Passed parameters checking function
        E23.stypy_localization = localization
        E23.stypy_type_of_self = None
        E23.stypy_type_store = module_type_store
        E23.stypy_function_name = 'E23'
        E23.stypy_param_names_list = ['h2', 'k2', 's']
        E23.stypy_varargs_param_name = None
        E23.stypy_kwargs_param_name = None
        E23.stypy_call_defaults = defaults
        E23.stypy_call_varargs = varargs
        E23.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E23', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E23', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E23(...)' code ##################

        # Getting the type of 's' (line 211)
        s_539071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 's')
        
        # Call to sqrt(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to abs(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 's' (line 211)
        s_539074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 28), 's', False)
        # Getting the type of 's' (line 211)
        s_539075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 30), 's', False)
        # Applying the binary operator '*' (line 211)
        result_mul_539076 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 28), '*', s_539074, s_539075)
        
        # Getting the type of 'h2' (line 211)
        h2_539077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 34), 'h2', False)
        # Applying the binary operator '-' (line 211)
        result_sub_539078 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 28), '-', result_mul_539076, h2_539077)
        
        # Processing the call keyword arguments (line 211)
        kwargs_539079 = {}
        # Getting the type of 'abs' (line 211)
        abs_539073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'abs', False)
        # Calling abs(args, kwargs) (line 211)
        abs_call_result_539080 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), abs_539073, *[result_sub_539078], **kwargs_539079)
        
        # Processing the call keyword arguments (line 211)
        kwargs_539081 = {}
        # Getting the type of 'sqrt' (line 211)
        sqrt_539072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 211)
        sqrt_call_result_539082 = invoke(stypy.reporting.localization.Localization(__file__, 211, 19), sqrt_539072, *[abs_call_result_539080], **kwargs_539081)
        
        # Applying the binary operator '*' (line 211)
        result_mul_539083 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 15), '*', s_539071, sqrt_call_result_539082)
        
        # Assigning a type to the variable 'stypy_return_type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', result_mul_539083)
        
        # ################# End of 'E23(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E23' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_539084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539084)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E23'
        return stypy_return_type_539084

    # Assigning a type to the variable 'E23' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'E23', E23)

    @norecursion
    def E24(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E24'
        module_type_store = module_type_store.open_function_context('E24', 213, 4, False)
        
        # Passed parameters checking function
        E24.stypy_localization = localization
        E24.stypy_type_of_self = None
        E24.stypy_type_store = module_type_store
        E24.stypy_function_name = 'E24'
        E24.stypy_param_names_list = ['h2', 'k2', 's']
        E24.stypy_varargs_param_name = None
        E24.stypy_kwargs_param_name = None
        E24.stypy_call_defaults = defaults
        E24.stypy_call_varargs = varargs
        E24.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E24', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E24', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E24(...)' code ##################

        # Getting the type of 's' (line 214)
        s_539085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 's')
        
        # Call to sqrt(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to abs(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 's' (line 214)
        s_539088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 's', False)
        # Getting the type of 's' (line 214)
        s_539089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 's', False)
        # Applying the binary operator '*' (line 214)
        result_mul_539090 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 28), '*', s_539088, s_539089)
        
        # Getting the type of 'k2' (line 214)
        k2_539091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'k2', False)
        # Applying the binary operator '-' (line 214)
        result_sub_539092 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 28), '-', result_mul_539090, k2_539091)
        
        # Processing the call keyword arguments (line 214)
        kwargs_539093 = {}
        # Getting the type of 'abs' (line 214)
        abs_539087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'abs', False)
        # Calling abs(args, kwargs) (line 214)
        abs_call_result_539094 = invoke(stypy.reporting.localization.Localization(__file__, 214, 24), abs_539087, *[result_sub_539092], **kwargs_539093)
        
        # Processing the call keyword arguments (line 214)
        kwargs_539095 = {}
        # Getting the type of 'sqrt' (line 214)
        sqrt_539086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 214)
        sqrt_call_result_539096 = invoke(stypy.reporting.localization.Localization(__file__, 214, 19), sqrt_539086, *[abs_call_result_539094], **kwargs_539095)
        
        # Applying the binary operator '*' (line 214)
        result_mul_539097 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 15), '*', s_539085, sqrt_call_result_539096)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', result_mul_539097)
        
        # ################# End of 'E24(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E24' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_539098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E24'
        return stypy_return_type_539098

    # Assigning a type to the variable 'E24' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'E24', E24)

    @norecursion
    def E25(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E25'
        module_type_store = module_type_store.open_function_context('E25', 216, 4, False)
        
        # Passed parameters checking function
        E25.stypy_localization = localization
        E25.stypy_type_of_self = None
        E25.stypy_type_store = module_type_store
        E25.stypy_function_name = 'E25'
        E25.stypy_param_names_list = ['h2', 'k2', 's']
        E25.stypy_varargs_param_name = None
        E25.stypy_kwargs_param_name = None
        E25.stypy_call_defaults = defaults
        E25.stypy_call_varargs = varargs
        E25.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E25', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E25', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E25(...)' code ##################

        
        # Call to sqrt(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to abs(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 's' (line 217)
        s_539101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 's', False)
        # Getting the type of 's' (line 217)
        s_539102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 's', False)
        # Applying the binary operator '*' (line 217)
        result_mul_539103 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 25), '*', s_539101, s_539102)
        
        # Getting the type of 'h2' (line 217)
        h2_539104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'h2', False)
        # Applying the binary operator '-' (line 217)
        result_sub_539105 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 25), '-', result_mul_539103, h2_539104)
        
        # Getting the type of 's' (line 217)
        s_539106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 's', False)
        # Getting the type of 's' (line 217)
        s_539107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 38), 's', False)
        # Applying the binary operator '*' (line 217)
        result_mul_539108 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 36), '*', s_539106, s_539107)
        
        # Getting the type of 'k2' (line 217)
        k2_539109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'k2', False)
        # Applying the binary operator '-' (line 217)
        result_sub_539110 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 36), '-', result_mul_539108, k2_539109)
        
        # Applying the binary operator '*' (line 217)
        result_mul_539111 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 24), '*', result_sub_539105, result_sub_539110)
        
        # Processing the call keyword arguments (line 217)
        kwargs_539112 = {}
        # Getting the type of 'abs' (line 217)
        abs_539100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 217)
        abs_call_result_539113 = invoke(stypy.reporting.localization.Localization(__file__, 217, 20), abs_539100, *[result_mul_539111], **kwargs_539112)
        
        # Processing the call keyword arguments (line 217)
        kwargs_539114 = {}
        # Getting the type of 'sqrt' (line 217)
        sqrt_539099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 217)
        sqrt_call_result_539115 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), sqrt_539099, *[abs_call_result_539113], **kwargs_539114)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', sqrt_call_result_539115)
        
        # ################# End of 'E25(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E25' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_539116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539116)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E25'
        return stypy_return_type_539116

    # Assigning a type to the variable 'E25' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'E25', E25)

    @norecursion
    def E31(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E31'
        module_type_store = module_type_store.open_function_context('E31', 219, 4, False)
        
        # Passed parameters checking function
        E31.stypy_localization = localization
        E31.stypy_type_of_self = None
        E31.stypy_type_store = module_type_store
        E31.stypy_function_name = 'E31'
        E31.stypy_param_names_list = ['h2', 'k2', 's']
        E31.stypy_varargs_param_name = None
        E31.stypy_kwargs_param_name = None
        E31.stypy_call_defaults = defaults
        E31.stypy_call_varargs = varargs
        E31.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E31', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E31', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E31(...)' code ##################

        # Getting the type of 's' (line 220)
        s_539117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 's')
        # Getting the type of 's' (line 220)
        s_539118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 's')
        # Applying the binary operator '*' (line 220)
        result_mul_539119 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 15), '*', s_539117, s_539118)
        
        # Getting the type of 's' (line 220)
        s_539120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 's')
        # Applying the binary operator '*' (line 220)
        result_mul_539121 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 18), '*', result_mul_539119, s_539120)
        
        # Getting the type of 's' (line 220)
        s_539122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 's')
        int_539123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 26), 'int')
        # Applying the binary operator 'div' (line 220)
        result_div_539124 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 24), 'div', s_539122, int_539123)
        
        int_539125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 30), 'int')
        # Getting the type of 'h2' (line 220)
        h2_539126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'h2')
        # Getting the type of 'k2' (line 220)
        k2_539127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'k2')
        # Applying the binary operator '+' (line 220)
        result_add_539128 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 33), '+', h2_539126, k2_539127)
        
        # Applying the binary operator '*' (line 220)
        result_mul_539129 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 30), '*', int_539125, result_add_539128)
        
        
        # Call to sqrt(...): (line 220)
        # Processing the call arguments (line 220)
        int_539131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 49), 'int')
        # Getting the type of 'h2' (line 220)
        h2_539132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 52), 'h2', False)
        # Getting the type of 'k2' (line 220)
        k2_539133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 57), 'k2', False)
        # Applying the binary operator '+' (line 220)
        result_add_539134 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 52), '+', h2_539132, k2_539133)
        
        # Applying the binary operator '*' (line 220)
        result_mul_539135 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 49), '*', int_539131, result_add_539134)
        
        # Getting the type of 'h2' (line 220)
        h2_539136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 62), 'h2', False)
        # Getting the type of 'k2' (line 220)
        k2_539137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 67), 'k2', False)
        # Applying the binary operator '+' (line 220)
        result_add_539138 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 62), '+', h2_539136, k2_539137)
        
        # Applying the binary operator '*' (line 220)
        result_mul_539139 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 60), '*', result_mul_539135, result_add_539138)
        
        int_539140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
        # Getting the type of 'h2' (line 221)
        h2_539141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'h2', False)
        # Applying the binary operator '*' (line 221)
        result_mul_539142 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 8), '*', int_539140, h2_539141)
        
        # Getting the type of 'k2' (line 221)
        k2_539143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'k2', False)
        # Applying the binary operator '*' (line 221)
        result_mul_539144 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 13), '*', result_mul_539142, k2_539143)
        
        # Applying the binary operator '-' (line 220)
        result_sub_539145 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 49), '-', result_mul_539139, result_mul_539144)
        
        # Processing the call keyword arguments (line 220)
        kwargs_539146 = {}
        # Getting the type of 'sqrt' (line 220)
        sqrt_539130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 220)
        sqrt_call_result_539147 = invoke(stypy.reporting.localization.Localization(__file__, 220, 44), sqrt_539130, *[result_sub_539145], **kwargs_539146)
        
        # Applying the binary operator '+' (line 220)
        result_add_539148 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 30), '+', result_mul_539129, sqrt_call_result_539147)
        
        # Applying the binary operator '*' (line 220)
        result_mul_539149 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 23), '*', result_div_539124, result_add_539148)
        
        # Applying the binary operator '-' (line 220)
        result_sub_539150 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 15), '-', result_mul_539121, result_mul_539149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', result_sub_539150)
        
        # ################# End of 'E31(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E31' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_539151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E31'
        return stypy_return_type_539151

    # Assigning a type to the variable 'E31' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'E31', E31)

    @norecursion
    def E32(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E32'
        module_type_store = module_type_store.open_function_context('E32', 223, 4, False)
        
        # Passed parameters checking function
        E32.stypy_localization = localization
        E32.stypy_type_of_self = None
        E32.stypy_type_store = module_type_store
        E32.stypy_function_name = 'E32'
        E32.stypy_param_names_list = ['h2', 'k2', 's']
        E32.stypy_varargs_param_name = None
        E32.stypy_kwargs_param_name = None
        E32.stypy_call_defaults = defaults
        E32.stypy_call_varargs = varargs
        E32.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E32', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E32', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E32(...)' code ##################

        # Getting the type of 's' (line 224)
        s_539152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 's')
        # Getting the type of 's' (line 224)
        s_539153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 's')
        # Applying the binary operator '*' (line 224)
        result_mul_539154 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 15), '*', s_539152, s_539153)
        
        # Getting the type of 's' (line 224)
        s_539155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 's')
        # Applying the binary operator '*' (line 224)
        result_mul_539156 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 18), '*', result_mul_539154, s_539155)
        
        # Getting the type of 's' (line 224)
        s_539157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 's')
        int_539158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 26), 'int')
        # Applying the binary operator 'div' (line 224)
        result_div_539159 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 24), 'div', s_539157, int_539158)
        
        int_539160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 30), 'int')
        # Getting the type of 'h2' (line 224)
        h2_539161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'h2')
        # Getting the type of 'k2' (line 224)
        k2_539162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'k2')
        # Applying the binary operator '+' (line 224)
        result_add_539163 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 33), '+', h2_539161, k2_539162)
        
        # Applying the binary operator '*' (line 224)
        result_mul_539164 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 30), '*', int_539160, result_add_539163)
        
        
        # Call to sqrt(...): (line 224)
        # Processing the call arguments (line 224)
        int_539166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'int')
        # Getting the type of 'h2' (line 224)
        h2_539167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 52), 'h2', False)
        # Getting the type of 'k2' (line 224)
        k2_539168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'k2', False)
        # Applying the binary operator '+' (line 224)
        result_add_539169 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 52), '+', h2_539167, k2_539168)
        
        # Applying the binary operator '*' (line 224)
        result_mul_539170 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 49), '*', int_539166, result_add_539169)
        
        # Getting the type of 'h2' (line 224)
        h2_539171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 62), 'h2', False)
        # Getting the type of 'k2' (line 224)
        k2_539172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 67), 'k2', False)
        # Applying the binary operator '+' (line 224)
        result_add_539173 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 62), '+', h2_539171, k2_539172)
        
        # Applying the binary operator '*' (line 224)
        result_mul_539174 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 60), '*', result_mul_539170, result_add_539173)
        
        int_539175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'int')
        # Getting the type of 'h2' (line 225)
        h2_539176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'h2', False)
        # Applying the binary operator '*' (line 225)
        result_mul_539177 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 8), '*', int_539175, h2_539176)
        
        # Getting the type of 'k2' (line 225)
        k2_539178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 14), 'k2', False)
        # Applying the binary operator '*' (line 225)
        result_mul_539179 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 13), '*', result_mul_539177, k2_539178)
        
        # Applying the binary operator '-' (line 224)
        result_sub_539180 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 49), '-', result_mul_539174, result_mul_539179)
        
        # Processing the call keyword arguments (line 224)
        kwargs_539181 = {}
        # Getting the type of 'sqrt' (line 224)
        sqrt_539165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 44), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 224)
        sqrt_call_result_539182 = invoke(stypy.reporting.localization.Localization(__file__, 224, 44), sqrt_539165, *[result_sub_539180], **kwargs_539181)
        
        # Applying the binary operator '-' (line 224)
        result_sub_539183 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 30), '-', result_mul_539164, sqrt_call_result_539182)
        
        # Applying the binary operator '*' (line 224)
        result_mul_539184 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 23), '*', result_div_539159, result_sub_539183)
        
        # Applying the binary operator '-' (line 224)
        result_sub_539185 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 15), '-', result_mul_539156, result_mul_539184)
        
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', result_sub_539185)
        
        # ################# End of 'E32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E32' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_539186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539186)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E32'
        return stypy_return_type_539186

    # Assigning a type to the variable 'E32' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'E32', E32)

    @norecursion
    def E33(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E33'
        module_type_store = module_type_store.open_function_context('E33', 227, 4, False)
        
        # Passed parameters checking function
        E33.stypy_localization = localization
        E33.stypy_type_of_self = None
        E33.stypy_type_store = module_type_store
        E33.stypy_function_name = 'E33'
        E33.stypy_param_names_list = ['h2', 'k2', 's']
        E33.stypy_varargs_param_name = None
        E33.stypy_kwargs_param_name = None
        E33.stypy_call_defaults = defaults
        E33.stypy_call_varargs = varargs
        E33.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E33', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E33', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E33(...)' code ##################

        
        # Call to sqrt(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to abs(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 's' (line 228)
        s_539189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 's', False)
        # Getting the type of 's' (line 228)
        s_539190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 's', False)
        # Applying the binary operator '*' (line 228)
        result_mul_539191 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 24), '*', s_539189, s_539190)
        
        # Getting the type of 'h2' (line 228)
        h2_539192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'h2', False)
        # Applying the binary operator '-' (line 228)
        result_sub_539193 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 24), '-', result_mul_539191, h2_539192)
        
        # Processing the call keyword arguments (line 228)
        kwargs_539194 = {}
        # Getting the type of 'abs' (line 228)
        abs_539188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 228)
        abs_call_result_539195 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), abs_539188, *[result_sub_539193], **kwargs_539194)
        
        # Processing the call keyword arguments (line 228)
        kwargs_539196 = {}
        # Getting the type of 'sqrt' (line 228)
        sqrt_539187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 228)
        sqrt_call_result_539197 = invoke(stypy.reporting.localization.Localization(__file__, 228, 15), sqrt_539187, *[abs_call_result_539195], **kwargs_539196)
        
        # Getting the type of 's' (line 228)
        s_539198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 36), 's')
        # Getting the type of 's' (line 228)
        s_539199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 38), 's')
        # Applying the binary operator '*' (line 228)
        result_mul_539200 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 36), '*', s_539198, s_539199)
        
        int_539201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 42), 'int')
        int_539202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 44), 'int')
        # Applying the binary operator 'div' (line 228)
        result_div_539203 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 42), 'div', int_539201, int_539202)
        
        # Getting the type of 'h2' (line 228)
        h2_539204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 48), 'h2')
        int_539205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 53), 'int')
        # Getting the type of 'k2' (line 228)
        k2_539206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 55), 'k2')
        # Applying the binary operator '*' (line 228)
        result_mul_539207 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 53), '*', int_539205, k2_539206)
        
        # Applying the binary operator '+' (line 228)
        result_add_539208 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 48), '+', h2_539204, result_mul_539207)
        
        
        # Call to sqrt(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Call to abs(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'h2' (line 228)
        h2_539211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 71), 'h2', False)
        int_539212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 8), 'int')
        # Getting the type of 'k2' (line 229)
        k2_539213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 10), 'k2', False)
        # Applying the binary operator '*' (line 229)
        result_mul_539214 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 8), '*', int_539212, k2_539213)
        
        # Applying the binary operator '+' (line 228)
        result_add_539215 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 71), '+', h2_539211, result_mul_539214)
        
        # Getting the type of 'h2' (line 229)
        h2_539216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'h2', False)
        int_539217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 20), 'int')
        # Getting the type of 'k2' (line 229)
        k2_539218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'k2', False)
        # Applying the binary operator '*' (line 229)
        result_mul_539219 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '*', int_539217, k2_539218)
        
        # Applying the binary operator '+' (line 229)
        result_add_539220 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '+', h2_539216, result_mul_539219)
        
        # Applying the binary operator '*' (line 228)
        result_mul_539221 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 70), '*', result_add_539215, result_add_539220)
        
        int_539222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'int')
        # Getting the type of 'h2' (line 229)
        h2_539223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'h2', False)
        # Applying the binary operator '*' (line 229)
        result_mul_539224 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 28), '*', int_539222, h2_539223)
        
        # Getting the type of 'k2' (line 229)
        k2_539225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 33), 'k2', False)
        # Applying the binary operator '*' (line 229)
        result_mul_539226 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 32), '*', result_mul_539224, k2_539225)
        
        # Applying the binary operator '-' (line 228)
        result_sub_539227 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 70), '-', result_mul_539221, result_mul_539226)
        
        # Processing the call keyword arguments (line 228)
        kwargs_539228 = {}
        # Getting the type of 'abs' (line 228)
        abs_539210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 66), 'abs', False)
        # Calling abs(args, kwargs) (line 228)
        abs_call_result_539229 = invoke(stypy.reporting.localization.Localization(__file__, 228, 66), abs_539210, *[result_sub_539227], **kwargs_539228)
        
        # Processing the call keyword arguments (line 228)
        kwargs_539230 = {}
        # Getting the type of 'sqrt' (line 228)
        sqrt_539209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 61), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 228)
        sqrt_call_result_539231 = invoke(stypy.reporting.localization.Localization(__file__, 228, 61), sqrt_539209, *[abs_call_result_539229], **kwargs_539230)
        
        # Applying the binary operator '+' (line 228)
        result_add_539232 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 47), '+', result_add_539208, sqrt_call_result_539231)
        
        # Applying the binary operator '*' (line 228)
        result_mul_539233 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 45), '*', result_div_539203, result_add_539232)
        
        # Applying the binary operator '-' (line 228)
        result_sub_539234 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 36), '-', result_mul_539200, result_mul_539233)
        
        # Applying the binary operator '*' (line 228)
        result_mul_539235 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 15), '*', sqrt_call_result_539197, result_sub_539234)
        
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', result_mul_539235)
        
        # ################# End of 'E33(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E33' in the type store
        # Getting the type of 'stypy_return_type' (line 227)
        stypy_return_type_539236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E33'
        return stypy_return_type_539236

    # Assigning a type to the variable 'E33' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'E33', E33)

    @norecursion
    def E34(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E34'
        module_type_store = module_type_store.open_function_context('E34', 231, 4, False)
        
        # Passed parameters checking function
        E34.stypy_localization = localization
        E34.stypy_type_of_self = None
        E34.stypy_type_store = module_type_store
        E34.stypy_function_name = 'E34'
        E34.stypy_param_names_list = ['h2', 'k2', 's']
        E34.stypy_varargs_param_name = None
        E34.stypy_kwargs_param_name = None
        E34.stypy_call_defaults = defaults
        E34.stypy_call_varargs = varargs
        E34.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E34', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E34', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E34(...)' code ##################

        
        # Call to sqrt(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to abs(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 's' (line 232)
        s_539239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 's', False)
        # Getting the type of 's' (line 232)
        s_539240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 's', False)
        # Applying the binary operator '*' (line 232)
        result_mul_539241 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 24), '*', s_539239, s_539240)
        
        # Getting the type of 'h2' (line 232)
        h2_539242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'h2', False)
        # Applying the binary operator '-' (line 232)
        result_sub_539243 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 24), '-', result_mul_539241, h2_539242)
        
        # Processing the call keyword arguments (line 232)
        kwargs_539244 = {}
        # Getting the type of 'abs' (line 232)
        abs_539238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 232)
        abs_call_result_539245 = invoke(stypy.reporting.localization.Localization(__file__, 232, 20), abs_539238, *[result_sub_539243], **kwargs_539244)
        
        # Processing the call keyword arguments (line 232)
        kwargs_539246 = {}
        # Getting the type of 'sqrt' (line 232)
        sqrt_539237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 232)
        sqrt_call_result_539247 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), sqrt_539237, *[abs_call_result_539245], **kwargs_539246)
        
        # Getting the type of 's' (line 232)
        s_539248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 's')
        # Getting the type of 's' (line 232)
        s_539249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 's')
        # Applying the binary operator '*' (line 232)
        result_mul_539250 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 36), '*', s_539248, s_539249)
        
        int_539251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 42), 'int')
        int_539252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 44), 'int')
        # Applying the binary operator 'div' (line 232)
        result_div_539253 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 42), 'div', int_539251, int_539252)
        
        # Getting the type of 'h2' (line 232)
        h2_539254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 48), 'h2')
        int_539255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 53), 'int')
        # Getting the type of 'k2' (line 232)
        k2_539256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 55), 'k2')
        # Applying the binary operator '*' (line 232)
        result_mul_539257 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 53), '*', int_539255, k2_539256)
        
        # Applying the binary operator '+' (line 232)
        result_add_539258 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 48), '+', h2_539254, result_mul_539257)
        
        
        # Call to sqrt(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to abs(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'h2' (line 232)
        h2_539261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 71), 'h2', False)
        int_539262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 8), 'int')
        # Getting the type of 'k2' (line 233)
        k2_539263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 10), 'k2', False)
        # Applying the binary operator '*' (line 233)
        result_mul_539264 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 8), '*', int_539262, k2_539263)
        
        # Applying the binary operator '+' (line 232)
        result_add_539265 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 71), '+', h2_539261, result_mul_539264)
        
        # Getting the type of 'h2' (line 233)
        h2_539266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'h2', False)
        int_539267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'int')
        # Getting the type of 'k2' (line 233)
        k2_539268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'k2', False)
        # Applying the binary operator '*' (line 233)
        result_mul_539269 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 20), '*', int_539267, k2_539268)
        
        # Applying the binary operator '+' (line 233)
        result_add_539270 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), '+', h2_539266, result_mul_539269)
        
        # Applying the binary operator '*' (line 232)
        result_mul_539271 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 70), '*', result_add_539265, result_add_539270)
        
        int_539272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'int')
        # Getting the type of 'h2' (line 233)
        h2_539273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'h2', False)
        # Applying the binary operator '*' (line 233)
        result_mul_539274 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 28), '*', int_539272, h2_539273)
        
        # Getting the type of 'k2' (line 233)
        k2_539275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 33), 'k2', False)
        # Applying the binary operator '*' (line 233)
        result_mul_539276 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 32), '*', result_mul_539274, k2_539275)
        
        # Applying the binary operator '-' (line 232)
        result_sub_539277 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 70), '-', result_mul_539271, result_mul_539276)
        
        # Processing the call keyword arguments (line 232)
        kwargs_539278 = {}
        # Getting the type of 'abs' (line 232)
        abs_539260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 66), 'abs', False)
        # Calling abs(args, kwargs) (line 232)
        abs_call_result_539279 = invoke(stypy.reporting.localization.Localization(__file__, 232, 66), abs_539260, *[result_sub_539277], **kwargs_539278)
        
        # Processing the call keyword arguments (line 232)
        kwargs_539280 = {}
        # Getting the type of 'sqrt' (line 232)
        sqrt_539259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 61), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 232)
        sqrt_call_result_539281 = invoke(stypy.reporting.localization.Localization(__file__, 232, 61), sqrt_539259, *[abs_call_result_539279], **kwargs_539280)
        
        # Applying the binary operator '-' (line 232)
        result_sub_539282 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 47), '-', result_add_539258, sqrt_call_result_539281)
        
        # Applying the binary operator '*' (line 232)
        result_mul_539283 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 45), '*', result_div_539253, result_sub_539282)
        
        # Applying the binary operator '-' (line 232)
        result_sub_539284 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 36), '-', result_mul_539250, result_mul_539283)
        
        # Applying the binary operator '*' (line 232)
        result_mul_539285 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 15), '*', sqrt_call_result_539247, result_sub_539284)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', result_mul_539285)
        
        # ################# End of 'E34(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E34' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_539286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E34'
        return stypy_return_type_539286

    # Assigning a type to the variable 'E34' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'E34', E34)

    @norecursion
    def E35(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E35'
        module_type_store = module_type_store.open_function_context('E35', 235, 4, False)
        
        # Passed parameters checking function
        E35.stypy_localization = localization
        E35.stypy_type_of_self = None
        E35.stypy_type_store = module_type_store
        E35.stypy_function_name = 'E35'
        E35.stypy_param_names_list = ['h2', 'k2', 's']
        E35.stypy_varargs_param_name = None
        E35.stypy_kwargs_param_name = None
        E35.stypy_call_defaults = defaults
        E35.stypy_call_varargs = varargs
        E35.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E35', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E35', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E35(...)' code ##################

        
        # Call to sqrt(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to abs(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 's' (line 236)
        s_539289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 's', False)
        # Getting the type of 's' (line 236)
        s_539290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 26), 's', False)
        # Applying the binary operator '*' (line 236)
        result_mul_539291 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 24), '*', s_539289, s_539290)
        
        # Getting the type of 'k2' (line 236)
        k2_539292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), 'k2', False)
        # Applying the binary operator '-' (line 236)
        result_sub_539293 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 24), '-', result_mul_539291, k2_539292)
        
        # Processing the call keyword arguments (line 236)
        kwargs_539294 = {}
        # Getting the type of 'abs' (line 236)
        abs_539288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 236)
        abs_call_result_539295 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), abs_539288, *[result_sub_539293], **kwargs_539294)
        
        # Processing the call keyword arguments (line 236)
        kwargs_539296 = {}
        # Getting the type of 'sqrt' (line 236)
        sqrt_539287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 236)
        sqrt_call_result_539297 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), sqrt_539287, *[abs_call_result_539295], **kwargs_539296)
        
        # Getting the type of 's' (line 236)
        s_539298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 's')
        # Getting the type of 's' (line 236)
        s_539299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 38), 's')
        # Applying the binary operator '*' (line 236)
        result_mul_539300 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 36), '*', s_539298, s_539299)
        
        int_539301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 42), 'int')
        int_539302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 44), 'int')
        # Applying the binary operator 'div' (line 236)
        result_div_539303 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 42), 'div', int_539301, int_539302)
        
        int_539304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 48), 'int')
        # Getting the type of 'h2' (line 236)
        h2_539305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 50), 'h2')
        # Applying the binary operator '*' (line 236)
        result_mul_539306 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 48), '*', int_539304, h2_539305)
        
        # Getting the type of 'k2' (line 236)
        k2_539307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 55), 'k2')
        # Applying the binary operator '+' (line 236)
        result_add_539308 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 48), '+', result_mul_539306, k2_539307)
        
        
        # Call to sqrt(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to abs(...): (line 236)
        # Processing the call arguments (line 236)
        int_539311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 71), 'int')
        # Getting the type of 'h2' (line 236)
        h2_539312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 73), 'h2', False)
        # Applying the binary operator '*' (line 236)
        result_mul_539313 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 71), '*', int_539311, h2_539312)
        
        # Getting the type of 'k2' (line 237)
        k2_539314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 10), 'k2', False)
        # Applying the binary operator '+' (line 236)
        result_add_539315 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 71), '+', result_mul_539313, k2_539314)
        
        int_539316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 15), 'int')
        # Getting the type of 'h2' (line 237)
        h2_539317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'h2', False)
        # Applying the binary operator '*' (line 237)
        result_mul_539318 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 15), '*', int_539316, h2_539317)
        
        # Getting the type of 'k2' (line 237)
        k2_539319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'k2', False)
        # Applying the binary operator '+' (line 237)
        result_add_539320 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 15), '+', result_mul_539318, k2_539319)
        
        # Applying the binary operator '*' (line 236)
        result_mul_539321 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 70), '*', result_add_539315, result_add_539320)
        
        int_539322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 28), 'int')
        # Getting the type of 'h2' (line 237)
        h2_539323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 30), 'h2', False)
        # Applying the binary operator '*' (line 237)
        result_mul_539324 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 28), '*', int_539322, h2_539323)
        
        # Getting the type of 'k2' (line 237)
        k2_539325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'k2', False)
        # Applying the binary operator '*' (line 237)
        result_mul_539326 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 32), '*', result_mul_539324, k2_539325)
        
        # Applying the binary operator '-' (line 236)
        result_sub_539327 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 70), '-', result_mul_539321, result_mul_539326)
        
        # Processing the call keyword arguments (line 236)
        kwargs_539328 = {}
        # Getting the type of 'abs' (line 236)
        abs_539310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 66), 'abs', False)
        # Calling abs(args, kwargs) (line 236)
        abs_call_result_539329 = invoke(stypy.reporting.localization.Localization(__file__, 236, 66), abs_539310, *[result_sub_539327], **kwargs_539328)
        
        # Processing the call keyword arguments (line 236)
        kwargs_539330 = {}
        # Getting the type of 'sqrt' (line 236)
        sqrt_539309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 61), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 236)
        sqrt_call_result_539331 = invoke(stypy.reporting.localization.Localization(__file__, 236, 61), sqrt_539309, *[abs_call_result_539329], **kwargs_539330)
        
        # Applying the binary operator '+' (line 236)
        result_add_539332 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 47), '+', result_add_539308, sqrt_call_result_539331)
        
        # Applying the binary operator '*' (line 236)
        result_mul_539333 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 45), '*', result_div_539303, result_add_539332)
        
        # Applying the binary operator '-' (line 236)
        result_sub_539334 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 36), '-', result_mul_539300, result_mul_539333)
        
        # Applying the binary operator '*' (line 236)
        result_mul_539335 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 15), '*', sqrt_call_result_539297, result_sub_539334)
        
        # Assigning a type to the variable 'stypy_return_type' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', result_mul_539335)
        
        # ################# End of 'E35(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E35' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_539336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539336)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E35'
        return stypy_return_type_539336

    # Assigning a type to the variable 'E35' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'E35', E35)

    @norecursion
    def E36(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E36'
        module_type_store = module_type_store.open_function_context('E36', 239, 4, False)
        
        # Passed parameters checking function
        E36.stypy_localization = localization
        E36.stypy_type_of_self = None
        E36.stypy_type_store = module_type_store
        E36.stypy_function_name = 'E36'
        E36.stypy_param_names_list = ['h2', 'k2', 's']
        E36.stypy_varargs_param_name = None
        E36.stypy_kwargs_param_name = None
        E36.stypy_call_defaults = defaults
        E36.stypy_call_varargs = varargs
        E36.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E36', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E36', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E36(...)' code ##################

        
        # Call to sqrt(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Call to abs(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 's' (line 240)
        s_539339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 's', False)
        # Getting the type of 's' (line 240)
        s_539340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 's', False)
        # Applying the binary operator '*' (line 240)
        result_mul_539341 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 24), '*', s_539339, s_539340)
        
        # Getting the type of 'k2' (line 240)
        k2_539342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'k2', False)
        # Applying the binary operator '-' (line 240)
        result_sub_539343 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 24), '-', result_mul_539341, k2_539342)
        
        # Processing the call keyword arguments (line 240)
        kwargs_539344 = {}
        # Getting the type of 'abs' (line 240)
        abs_539338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 240)
        abs_call_result_539345 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), abs_539338, *[result_sub_539343], **kwargs_539344)
        
        # Processing the call keyword arguments (line 240)
        kwargs_539346 = {}
        # Getting the type of 'sqrt' (line 240)
        sqrt_539337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 240)
        sqrt_call_result_539347 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), sqrt_539337, *[abs_call_result_539345], **kwargs_539346)
        
        # Getting the type of 's' (line 240)
        s_539348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 36), 's')
        # Getting the type of 's' (line 240)
        s_539349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 38), 's')
        # Applying the binary operator '*' (line 240)
        result_mul_539350 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 36), '*', s_539348, s_539349)
        
        int_539351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 42), 'int')
        int_539352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 44), 'int')
        # Applying the binary operator 'div' (line 240)
        result_div_539353 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 42), 'div', int_539351, int_539352)
        
        int_539354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 48), 'int')
        # Getting the type of 'h2' (line 240)
        h2_539355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 50), 'h2')
        # Applying the binary operator '*' (line 240)
        result_mul_539356 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 48), '*', int_539354, h2_539355)
        
        # Getting the type of 'k2' (line 240)
        k2_539357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 55), 'k2')
        # Applying the binary operator '+' (line 240)
        result_add_539358 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 48), '+', result_mul_539356, k2_539357)
        
        
        # Call to sqrt(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Call to abs(...): (line 240)
        # Processing the call arguments (line 240)
        int_539361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 71), 'int')
        # Getting the type of 'h2' (line 240)
        h2_539362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 73), 'h2', False)
        # Applying the binary operator '*' (line 240)
        result_mul_539363 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 71), '*', int_539361, h2_539362)
        
        # Getting the type of 'k2' (line 241)
        k2_539364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 10), 'k2', False)
        # Applying the binary operator '+' (line 240)
        result_add_539365 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 71), '+', result_mul_539363, k2_539364)
        
        int_539366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 15), 'int')
        # Getting the type of 'h2' (line 241)
        h2_539367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'h2', False)
        # Applying the binary operator '*' (line 241)
        result_mul_539368 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '*', int_539366, h2_539367)
        
        # Getting the type of 'k2' (line 241)
        k2_539369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'k2', False)
        # Applying the binary operator '+' (line 241)
        result_add_539370 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 15), '+', result_mul_539368, k2_539369)
        
        # Applying the binary operator '*' (line 240)
        result_mul_539371 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 70), '*', result_add_539365, result_add_539370)
        
        int_539372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 28), 'int')
        # Getting the type of 'h2' (line 241)
        h2_539373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'h2', False)
        # Applying the binary operator '*' (line 241)
        result_mul_539374 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 28), '*', int_539372, h2_539373)
        
        # Getting the type of 'k2' (line 241)
        k2_539375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 33), 'k2', False)
        # Applying the binary operator '*' (line 241)
        result_mul_539376 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 32), '*', result_mul_539374, k2_539375)
        
        # Applying the binary operator '-' (line 240)
        result_sub_539377 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 70), '-', result_mul_539371, result_mul_539376)
        
        # Processing the call keyword arguments (line 240)
        kwargs_539378 = {}
        # Getting the type of 'abs' (line 240)
        abs_539360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 66), 'abs', False)
        # Calling abs(args, kwargs) (line 240)
        abs_call_result_539379 = invoke(stypy.reporting.localization.Localization(__file__, 240, 66), abs_539360, *[result_sub_539377], **kwargs_539378)
        
        # Processing the call keyword arguments (line 240)
        kwargs_539380 = {}
        # Getting the type of 'sqrt' (line 240)
        sqrt_539359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 61), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 240)
        sqrt_call_result_539381 = invoke(stypy.reporting.localization.Localization(__file__, 240, 61), sqrt_539359, *[abs_call_result_539379], **kwargs_539380)
        
        # Applying the binary operator '-' (line 240)
        result_sub_539382 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 47), '-', result_add_539358, sqrt_call_result_539381)
        
        # Applying the binary operator '*' (line 240)
        result_mul_539383 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 45), '*', result_div_539353, result_sub_539382)
        
        # Applying the binary operator '-' (line 240)
        result_sub_539384 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 36), '-', result_mul_539350, result_mul_539383)
        
        # Applying the binary operator '*' (line 240)
        result_mul_539385 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '*', sqrt_call_result_539347, result_sub_539384)
        
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', result_mul_539385)
        
        # ################# End of 'E36(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E36' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_539386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E36'
        return stypy_return_type_539386

    # Assigning a type to the variable 'E36' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'E36', E36)

    @norecursion
    def E37(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'E37'
        module_type_store = module_type_store.open_function_context('E37', 243, 4, False)
        
        # Passed parameters checking function
        E37.stypy_localization = localization
        E37.stypy_type_of_self = None
        E37.stypy_type_store = module_type_store
        E37.stypy_function_name = 'E37'
        E37.stypy_param_names_list = ['h2', 'k2', 's']
        E37.stypy_varargs_param_name = None
        E37.stypy_kwargs_param_name = None
        E37.stypy_call_defaults = defaults
        E37.stypy_call_varargs = varargs
        E37.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'E37', ['h2', 'k2', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'E37', localization, ['h2', 'k2', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'E37(...)' code ##################

        # Getting the type of 's' (line 244)
        s_539387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 's')
        
        # Call to sqrt(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to abs(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 's' (line 244)
        s_539390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 29), 's', False)
        # Getting the type of 's' (line 244)
        s_539391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 31), 's', False)
        # Applying the binary operator '*' (line 244)
        result_mul_539392 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 29), '*', s_539390, s_539391)
        
        # Getting the type of 'h2' (line 244)
        h2_539393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 35), 'h2', False)
        # Applying the binary operator '-' (line 244)
        result_sub_539394 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 29), '-', result_mul_539392, h2_539393)
        
        # Getting the type of 's' (line 244)
        s_539395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 40), 's', False)
        # Getting the type of 's' (line 244)
        s_539396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 42), 's', False)
        # Applying the binary operator '*' (line 244)
        result_mul_539397 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 40), '*', s_539395, s_539396)
        
        # Getting the type of 'k2' (line 244)
        k2_539398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 46), 'k2', False)
        # Applying the binary operator '-' (line 244)
        result_sub_539399 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 40), '-', result_mul_539397, k2_539398)
        
        # Applying the binary operator '*' (line 244)
        result_mul_539400 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 28), '*', result_sub_539394, result_sub_539399)
        
        # Processing the call keyword arguments (line 244)
        kwargs_539401 = {}
        # Getting the type of 'abs' (line 244)
        abs_539389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'abs', False)
        # Calling abs(args, kwargs) (line 244)
        abs_call_result_539402 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), abs_539389, *[result_mul_539400], **kwargs_539401)
        
        # Processing the call keyword arguments (line 244)
        kwargs_539403 = {}
        # Getting the type of 'sqrt' (line 244)
        sqrt_539388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 19), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 244)
        sqrt_call_result_539404 = invoke(stypy.reporting.localization.Localization(__file__, 244, 19), sqrt_539388, *[abs_call_result_539402], **kwargs_539403)
        
        # Applying the binary operator '*' (line 244)
        result_mul_539405 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 15), '*', s_539387, sqrt_call_result_539404)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', result_mul_539405)
        
        # ################# End of 'E37(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'E37' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_539406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'E37'
        return stypy_return_type_539406

    # Assigning a type to the variable 'E37' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'E37', E37)
    
    # Call to assert_equal(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Call to ellip_harm(...): (line 246)
    # Processing the call arguments (line 246)
    int_539409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 28), 'int')
    int_539410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 31), 'int')
    int_539411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 34), 'int')
    int_539412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 37), 'int')
    float_539413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 40), 'float')
    int_539414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 45), 'int')
    int_539415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 48), 'int')
    # Processing the call keyword arguments (line 246)
    kwargs_539416 = {}
    # Getting the type of 'ellip_harm' (line 246)
    ellip_harm_539408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'ellip_harm', False)
    # Calling ellip_harm(args, kwargs) (line 246)
    ellip_harm_call_result_539417 = invoke(stypy.reporting.localization.Localization(__file__, 246, 17), ellip_harm_539408, *[int_539409, int_539410, int_539411, int_539412, float_539413, int_539414, int_539415], **kwargs_539416)
    
    
    # Call to ellip_harm(...): (line 247)
    # Processing the call arguments (line 247)
    int_539419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'int')
    int_539420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 18), 'int')
    int_539421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'int')
    int_539422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 24), 'int')
    float_539423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 27), 'float')
    # Processing the call keyword arguments (line 247)
    kwargs_539424 = {}
    # Getting the type of 'ellip_harm' (line 247)
    ellip_harm_539418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'ellip_harm', False)
    # Calling ellip_harm(args, kwargs) (line 247)
    ellip_harm_call_result_539425 = invoke(stypy.reporting.localization.Localization(__file__, 247, 4), ellip_harm_539418, *[int_539419, int_539420, int_539421, int_539422, float_539423], **kwargs_539424)
    
    # Processing the call keyword arguments (line 246)
    kwargs_539426 = {}
    # Getting the type of 'assert_equal' (line 246)
    assert_equal_539407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 246)
    assert_equal_call_result_539427 = invoke(stypy.reporting.localization.Localization(__file__, 246, 4), assert_equal_539407, *[ellip_harm_call_result_539417, ellip_harm_call_result_539425], **kwargs_539426)
    
    
    # Assigning a Dict to a Name (line 249):
    
    # Assigning a Dict to a Name (line 249):
    
    # Obtaining an instance of the builtin type 'dict' (line 249)
    dict_539428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 18), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 249)
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_539429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    # Adding element type (line 249)
    int_539430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), tuple_539429, int_539430)
    # Adding element type (line 249)
    int_539431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 20), tuple_539429, int_539431)
    
    # Getting the type of 'E01' (line 249)
    E01_539432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 27), 'E01')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539429, E01_539432))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_539433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    # Adding element type (line 249)
    int_539434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 33), tuple_539433, int_539434)
    # Adding element type (line 249)
    int_539435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 33), tuple_539433, int_539435)
    
    # Getting the type of 'E11' (line 249)
    E11_539436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'E11')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539433, E11_539436))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_539437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    # Adding element type (line 249)
    int_539438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 46), tuple_539437, int_539438)
    # Adding element type (line 249)
    int_539439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 46), tuple_539437, int_539439)
    
    # Getting the type of 'E12' (line 249)
    E12_539440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 53), 'E12')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539437, E12_539440))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 249)
    tuple_539441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 249)
    # Adding element type (line 249)
    int_539442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 59), tuple_539441, int_539442)
    # Adding element type (line 249)
    int_539443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 59), tuple_539441, int_539443)
    
    # Getting the type of 'E13' (line 249)
    E13_539444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 66), 'E13')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539441, E13_539444))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_539445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    int_539446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), tuple_539445, int_539446)
    # Adding element type (line 250)
    int_539447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 20), tuple_539445, int_539447)
    
    # Getting the type of 'E21' (line 250)
    E21_539448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'E21')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539445, E21_539448))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_539449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    int_539450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 33), tuple_539449, int_539450)
    # Adding element type (line 250)
    int_539451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 33), tuple_539449, int_539451)
    
    # Getting the type of 'E22' (line 250)
    E22_539452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 40), 'E22')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539449, E22_539452))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_539453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    int_539454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 46), tuple_539453, int_539454)
    # Adding element type (line 250)
    int_539455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 46), tuple_539453, int_539455)
    
    # Getting the type of 'E23' (line 250)
    E23_539456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 53), 'E23')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539453, E23_539456))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 250)
    tuple_539457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 250)
    # Adding element type (line 250)
    int_539458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 59), tuple_539457, int_539458)
    # Adding element type (line 250)
    int_539459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 250, 59), tuple_539457, int_539459)
    
    # Getting the type of 'E24' (line 250)
    E24_539460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 66), 'E24')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539457, E24_539460))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_539461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    int_539462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 20), tuple_539461, int_539462)
    # Adding element type (line 251)
    int_539463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 20), tuple_539461, int_539463)
    
    # Getting the type of 'E25' (line 251)
    E25_539464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'E25')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539461, E25_539464))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_539465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    int_539466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 33), tuple_539465, int_539466)
    # Adding element type (line 251)
    int_539467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 33), tuple_539465, int_539467)
    
    # Getting the type of 'E31' (line 251)
    E31_539468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 40), 'E31')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539465, E31_539468))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_539469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    int_539470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 46), tuple_539469, int_539470)
    # Adding element type (line 251)
    int_539471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 46), tuple_539469, int_539471)
    
    # Getting the type of 'E32' (line 251)
    E32_539472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 53), 'E32')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539469, E32_539472))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_539473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    int_539474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 59), tuple_539473, int_539474)
    # Adding element type (line 251)
    int_539475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 59), tuple_539473, int_539475)
    
    # Getting the type of 'E33' (line 251)
    E33_539476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 66), 'E33')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539473, E33_539476))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 252)
    tuple_539477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 252)
    # Adding element type (line 252)
    int_539478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 20), tuple_539477, int_539478)
    # Adding element type (line 252)
    int_539479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 20), tuple_539477, int_539479)
    
    # Getting the type of 'E34' (line 252)
    E34_539480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'E34')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539477, E34_539480))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 252)
    tuple_539481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 252)
    # Adding element type (line 252)
    int_539482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 33), tuple_539481, int_539482)
    # Adding element type (line 252)
    int_539483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 33), tuple_539481, int_539483)
    
    # Getting the type of 'E35' (line 252)
    E35_539484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 40), 'E35')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539481, E35_539484))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 252)
    tuple_539485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 252)
    # Adding element type (line 252)
    int_539486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 46), tuple_539485, int_539486)
    # Adding element type (line 252)
    int_539487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 46), tuple_539485, int_539487)
    
    # Getting the type of 'E36' (line 252)
    E36_539488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 53), 'E36')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539485, E36_539488))
    # Adding element type (key, value) (line 249)
    
    # Obtaining an instance of the builtin type 'tuple' (line 252)
    tuple_539489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 252)
    # Adding element type (line 252)
    int_539490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 59), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 59), tuple_539489, int_539490)
    # Adding element type (line 252)
    int_539491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 62), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 59), tuple_539489, int_539491)
    
    # Getting the type of 'E37' (line 252)
    E37_539492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 66), 'E37')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 249, 18), dict_539428, (tuple_539489, E37_539492))
    
    # Assigning a type to the variable 'known_funcs' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'known_funcs', dict_539428)
    
    # Assigning a List to a Name (line 254):
    
    # Assigning a List to a Name (line 254):
    
    # Obtaining an instance of the builtin type 'list' (line 254)
    list_539493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 254)
    
    # Assigning a type to the variable 'point_ref' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'point_ref', list_539493)

    @norecursion
    def ellip_harm_known(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ellip_harm_known'
        module_type_store = module_type_store.open_function_context('ellip_harm_known', 256, 4, False)
        
        # Passed parameters checking function
        ellip_harm_known.stypy_localization = localization
        ellip_harm_known.stypy_type_of_self = None
        ellip_harm_known.stypy_type_store = module_type_store
        ellip_harm_known.stypy_function_name = 'ellip_harm_known'
        ellip_harm_known.stypy_param_names_list = ['h2', 'k2', 'n', 'p', 's']
        ellip_harm_known.stypy_varargs_param_name = None
        ellip_harm_known.stypy_kwargs_param_name = None
        ellip_harm_known.stypy_call_defaults = defaults
        ellip_harm_known.stypy_call_varargs = varargs
        ellip_harm_known.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'ellip_harm_known', ['h2', 'k2', 'n', 'p', 's'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ellip_harm_known', localization, ['h2', 'k2', 'n', 'p', 's'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ellip_harm_known(...)' code ##################

        
        
        # Call to range(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'h2' (line 257)
        h2_539495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'h2', False)
        # Obtaining the member 'size' of a type (line 257)
        size_539496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), h2_539495, 'size')
        # Processing the call keyword arguments (line 257)
        kwargs_539497 = {}
        # Getting the type of 'range' (line 257)
        range_539494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'range', False)
        # Calling range(args, kwargs) (line 257)
        range_call_result_539498 = invoke(stypy.reporting.localization.Localization(__file__, 257, 17), range_539494, *[size_539496], **kwargs_539497)
        
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 8), range_call_result_539498)
        # Getting the type of the for loop variable (line 257)
        for_loop_var_539499 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 8), range_call_result_539498)
        # Assigning a type to the variable 'i' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'i', for_loop_var_539499)
        # SSA begins for a for statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 258):
        
        # Assigning a Subscript to a Name (line 258):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_539500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        
        # Call to int(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 258)
        i_539502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'i', False)
        # Getting the type of 'n' (line 258)
        n_539503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'n', False)
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___539504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 36), n_539503, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_539505 = invoke(stypy.reporting.localization.Localization(__file__, 258, 36), getitem___539504, i_539502)
        
        # Processing the call keyword arguments (line 258)
        kwargs_539506 = {}
        # Getting the type of 'int' (line 258)
        int_539501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 32), 'int', False)
        # Calling int(args, kwargs) (line 258)
        int_call_result_539507 = invoke(stypy.reporting.localization.Localization(__file__, 258, 32), int_539501, *[subscript_call_result_539505], **kwargs_539506)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 32), tuple_539500, int_call_result_539507)
        # Adding element type (line 258)
        
        # Call to int(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 258)
        i_539509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 49), 'i', False)
        # Getting the type of 'p' (line 258)
        p_539510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 47), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___539511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 47), p_539510, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_539512 = invoke(stypy.reporting.localization.Localization(__file__, 258, 47), getitem___539511, i_539509)
        
        # Processing the call keyword arguments (line 258)
        kwargs_539513 = {}
        # Getting the type of 'int' (line 258)
        int_539508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 43), 'int', False)
        # Calling int(args, kwargs) (line 258)
        int_call_result_539514 = invoke(stypy.reporting.localization.Localization(__file__, 258, 43), int_539508, *[subscript_call_result_539512], **kwargs_539513)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 32), tuple_539500, int_call_result_539514)
        
        # Getting the type of 'known_funcs' (line 258)
        known_funcs_539515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'known_funcs')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___539516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), known_funcs_539515, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_539517 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), getitem___539516, tuple_539500)
        
        # Assigning a type to the variable 'func' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'func', subscript_call_result_539517)
        
        # Call to append(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Call to func(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 259)
        i_539521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'i', False)
        # Getting the type of 'h2' (line 259)
        h2_539522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'h2', False)
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___539523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 34), h2_539522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_539524 = invoke(stypy.reporting.localization.Localization(__file__, 259, 34), getitem___539523, i_539521)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 259)
        i_539525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 44), 'i', False)
        # Getting the type of 'k2' (line 259)
        k2_539526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'k2', False)
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___539527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 41), k2_539526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_539528 = invoke(stypy.reporting.localization.Localization(__file__, 259, 41), getitem___539527, i_539525)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 259)
        i_539529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'i', False)
        # Getting the type of 's' (line 259)
        s_539530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 48), 's', False)
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___539531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 48), s_539530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_539532 = invoke(stypy.reporting.localization.Localization(__file__, 259, 48), getitem___539531, i_539529)
        
        # Processing the call keyword arguments (line 259)
        kwargs_539533 = {}
        # Getting the type of 'func' (line 259)
        func_539520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 29), 'func', False)
        # Calling func(args, kwargs) (line 259)
        func_call_result_539534 = invoke(stypy.reporting.localization.Localization(__file__, 259, 29), func_539520, *[subscript_call_result_539524, subscript_call_result_539528, subscript_call_result_539532], **kwargs_539533)
        
        # Processing the call keyword arguments (line 259)
        kwargs_539535 = {}
        # Getting the type of 'point_ref' (line 259)
        point_ref_539518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'point_ref', False)
        # Obtaining the member 'append' of a type (line 259)
        append_539519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), point_ref_539518, 'append')
        # Calling append(args, kwargs) (line 259)
        append_call_result_539536 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), append_539519, *[func_call_result_539534], **kwargs_539535)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'point_ref' (line 260)
        point_ref_539537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'point_ref')
        # Assigning a type to the variable 'stypy_return_type' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type', point_ref_539537)
        
        # ################# End of 'ellip_harm_known(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ellip_harm_known' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_539538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ellip_harm_known'
        return stypy_return_type_539538

    # Assigning a type to the variable 'ellip_harm_known' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'ellip_harm_known', ellip_harm_known)
    
    # Call to seed(...): (line 262)
    # Processing the call arguments (line 262)
    int_539542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 19), 'int')
    # Processing the call keyword arguments (line 262)
    kwargs_539543 = {}
    # Getting the type of 'np' (line 262)
    np_539539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 262)
    random_539540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 4), np_539539, 'random')
    # Obtaining the member 'seed' of a type (line 262)
    seed_539541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 4), random_539540, 'seed')
    # Calling seed(args, kwargs) (line 262)
    seed_call_result_539544 = invoke(stypy.reporting.localization.Localization(__file__, 262, 4), seed_539541, *[int_539542], **kwargs_539543)
    
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to pareto(...): (line 263)
    # Processing the call arguments (line 263)
    float_539548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 26), 'float')
    # Processing the call keyword arguments (line 263)
    int_539549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 36), 'int')
    keyword_539550 = int_539549
    kwargs_539551 = {'size': keyword_539550}
    # Getting the type of 'np' (line 263)
    np_539545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 9), 'np', False)
    # Obtaining the member 'random' of a type (line 263)
    random_539546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 9), np_539545, 'random')
    # Obtaining the member 'pareto' of a type (line 263)
    pareto_539547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 9), random_539546, 'pareto')
    # Calling pareto(args, kwargs) (line 263)
    pareto_call_result_539552 = invoke(stypy.reporting.localization.Localization(__file__, 263, 9), pareto_539547, *[float_539548], **kwargs_539551)
    
    # Assigning a type to the variable 'h2' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'h2', pareto_call_result_539552)
    
    # Assigning a BinOp to a Name (line 264):
    
    # Assigning a BinOp to a Name (line 264):
    # Getting the type of 'h2' (line 264)
    h2_539553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 9), 'h2')
    int_539554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 13), 'int')
    
    # Call to pareto(...): (line 264)
    # Processing the call arguments (line 264)
    float_539558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 34), 'float')
    # Processing the call keyword arguments (line 264)
    # Getting the type of 'h2' (line 264)
    h2_539559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 44), 'h2', False)
    # Obtaining the member 'size' of a type (line 264)
    size_539560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 44), h2_539559, 'size')
    keyword_539561 = size_539560
    kwargs_539562 = {'size': keyword_539561}
    # Getting the type of 'np' (line 264)
    np_539555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'np', False)
    # Obtaining the member 'random' of a type (line 264)
    random_539556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), np_539555, 'random')
    # Obtaining the member 'pareto' of a type (line 264)
    pareto_539557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 17), random_539556, 'pareto')
    # Calling pareto(args, kwargs) (line 264)
    pareto_call_result_539563 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), pareto_539557, *[float_539558], **kwargs_539562)
    
    # Applying the binary operator '+' (line 264)
    result_add_539564 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 13), '+', int_539554, pareto_call_result_539563)
    
    # Applying the binary operator '*' (line 264)
    result_mul_539565 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 9), '*', h2_539553, result_add_539564)
    
    # Assigning a type to the variable 'k2' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'k2', result_mul_539565)
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to pareto(...): (line 265)
    # Processing the call arguments (line 265)
    float_539569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 25), 'float')
    # Processing the call keyword arguments (line 265)
    # Getting the type of 'h2' (line 265)
    h2_539570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 35), 'h2', False)
    # Obtaining the member 'size' of a type (line 265)
    size_539571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 35), h2_539570, 'size')
    keyword_539572 = size_539571
    kwargs_539573 = {'size': keyword_539572}
    # Getting the type of 'np' (line 265)
    np_539566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 265)
    random_539567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), np_539566, 'random')
    # Obtaining the member 'pareto' of a type (line 265)
    pareto_539568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), random_539567, 'pareto')
    # Calling pareto(args, kwargs) (line 265)
    pareto_call_result_539574 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), pareto_539568, *[float_539569], **kwargs_539573)
    
    # Assigning a type to the variable 's' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 's', pareto_call_result_539574)
    
    # Assigning a List to a Name (line 266):
    
    # Assigning a List to a Name (line 266):
    
    # Obtaining an instance of the builtin type 'list' (line 266)
    list_539575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 266)
    
    # Assigning a type to the variable 'points' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'points', list_539575)
    
    
    # Call to range(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'h2' (line 267)
    h2_539577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'h2', False)
    # Obtaining the member 'size' of a type (line 267)
    size_539578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 19), h2_539577, 'size')
    # Processing the call keyword arguments (line 267)
    kwargs_539579 = {}
    # Getting the type of 'range' (line 267)
    range_539576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 13), 'range', False)
    # Calling range(args, kwargs) (line 267)
    range_call_result_539580 = invoke(stypy.reporting.localization.Localization(__file__, 267, 13), range_539576, *[size_539578], **kwargs_539579)
    
    # Testing the type of a for loop iterable (line 267)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 4), range_call_result_539580)
    # Getting the type of the for loop variable (line 267)
    for_loop_var_539581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 4), range_call_result_539580)
    # Assigning a type to the variable 'i' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'i', for_loop_var_539581)
    # SSA begins for a for statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 268)
    # Processing the call arguments (line 268)
    int_539583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 23), 'int')
    # Processing the call keyword arguments (line 268)
    kwargs_539584 = {}
    # Getting the type of 'range' (line 268)
    range_539582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'range', False)
    # Calling range(args, kwargs) (line 268)
    range_call_result_539585 = invoke(stypy.reporting.localization.Localization(__file__, 268, 17), range_539582, *[int_539583], **kwargs_539584)
    
    # Testing the type of a for loop iterable (line 268)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 268, 8), range_call_result_539585)
    # Getting the type of the for loop variable (line 268)
    for_loop_var_539586 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 268, 8), range_call_result_539585)
    # Assigning a type to the variable 'n' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'n', for_loop_var_539586)
    # SSA begins for a for statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 269)
    # Processing the call arguments (line 269)
    int_539588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 27), 'int')
    int_539589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 30), 'int')
    # Getting the type of 'n' (line 269)
    n_539590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'n', False)
    # Applying the binary operator '*' (line 269)
    result_mul_539591 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 30), '*', int_539589, n_539590)
    
    int_539592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 34), 'int')
    # Applying the binary operator '+' (line 269)
    result_add_539593 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 30), '+', result_mul_539591, int_539592)
    
    # Processing the call keyword arguments (line 269)
    kwargs_539594 = {}
    # Getting the type of 'range' (line 269)
    range_539587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 'range', False)
    # Calling range(args, kwargs) (line 269)
    range_call_result_539595 = invoke(stypy.reporting.localization.Localization(__file__, 269, 21), range_539587, *[int_539588, result_add_539593], **kwargs_539594)
    
    # Testing the type of a for loop iterable (line 269)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 12), range_call_result_539595)
    # Getting the type of the for loop variable (line 269)
    for_loop_var_539596 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 12), range_call_result_539595)
    # Assigning a type to the variable 'p' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'p', for_loop_var_539596)
    # SSA begins for a for statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Obtaining an instance of the builtin type 'tuple' (line 270)
    tuple_539599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 270)
    # Adding element type (line 270)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 270)
    i_539600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'i', False)
    # Getting the type of 'h2' (line 270)
    h2_539601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 31), 'h2', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___539602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 31), h2_539601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_539603 = invoke(stypy.reporting.localization.Localization(__file__, 270, 31), getitem___539602, i_539600)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 31), tuple_539599, subscript_call_result_539603)
    # Adding element type (line 270)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 270)
    i_539604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 41), 'i', False)
    # Getting the type of 'k2' (line 270)
    k2_539605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'k2', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___539606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 38), k2_539605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_539607 = invoke(stypy.reporting.localization.Localization(__file__, 270, 38), getitem___539606, i_539604)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 31), tuple_539599, subscript_call_result_539607)
    # Adding element type (line 270)
    # Getting the type of 'n' (line 270)
    n_539608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 45), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 31), tuple_539599, n_539608)
    # Adding element type (line 270)
    # Getting the type of 'p' (line 270)
    p_539609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 48), 'p', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 31), tuple_539599, p_539609)
    # Adding element type (line 270)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 270)
    i_539610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 53), 'i', False)
    # Getting the type of 's' (line 270)
    s_539611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 51), 's', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___539612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 51), s_539611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_539613 = invoke(stypy.reporting.localization.Localization(__file__, 270, 51), getitem___539612, i_539610)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 31), tuple_539599, subscript_call_result_539613)
    
    # Processing the call keyword arguments (line 270)
    kwargs_539614 = {}
    # Getting the type of 'points' (line 270)
    points_539597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'points', False)
    # Obtaining the member 'append' of a type (line 270)
    append_539598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), points_539597, 'append')
    # Calling append(args, kwargs) (line 270)
    append_call_result_539615 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), append_539598, *[tuple_539599], **kwargs_539614)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to array(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'points' (line 271)
    points_539618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 22), 'points', False)
    # Processing the call keyword arguments (line 271)
    kwargs_539619 = {}
    # Getting the type of 'np' (line 271)
    np_539616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 271)
    array_539617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 13), np_539616, 'array')
    # Calling array(args, kwargs) (line 271)
    array_call_result_539620 = invoke(stypy.reporting.localization.Localization(__file__, 271, 13), array_539617, *[points_539618], **kwargs_539619)
    
    # Assigning a type to the variable 'points' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'points', array_call_result_539620)
    
    # Call to assert_func_equal(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'ellip_harm' (line 272)
    ellip_harm_539622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'ellip_harm', False)
    # Getting the type of 'ellip_harm_known' (line 272)
    ellip_harm_known_539623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 34), 'ellip_harm_known', False)
    # Getting the type of 'points' (line 272)
    points_539624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 52), 'points', False)
    # Processing the call keyword arguments (line 272)
    float_539625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 65), 'float')
    keyword_539626 = float_539625
    kwargs_539627 = {'rtol': keyword_539626}
    # Getting the type of 'assert_func_equal' (line 272)
    assert_func_equal_539621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'assert_func_equal', False)
    # Calling assert_func_equal(args, kwargs) (line 272)
    assert_func_equal_call_result_539628 = invoke(stypy.reporting.localization.Localization(__file__, 272, 4), assert_func_equal_539621, *[ellip_harm_539622, ellip_harm_known_539623, points_539624], **kwargs_539627)
    
    
    # ################# End of 'test_ellip_harm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ellip_harm' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_539629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_539629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ellip_harm'
    return stypy_return_type_539629

# Assigning a type to the variable 'test_ellip_harm' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'test_ellip_harm', test_ellip_harm)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
