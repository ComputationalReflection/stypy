
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import \
5:                           assert_array_almost_equal, assert_almost_equal, \
6:                           assert_allclose, assert_equal
7: 
8: import warnings
9: from scipy.signal import cont2discrete as c2d
10: from scipy.signal import dlsim, ss2tf, ss2zpk, lsim2, lti
11: 
12: # Author: Jeffrey Armstrong <jeff@approximatrix.com>
13: # March 29, 2011
14: 
15: 
16: class TestC2D(object):
17:     def test_zoh(self):
18:         ac = np.eye(2)
19:         bc = 0.5 * np.ones((2, 1))
20:         cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
21:         dc = np.array([[0.0], [0.0], [-0.33]])
22: 
23:         ad_truth = 1.648721270700128 * np.eye(2)
24:         bd_truth = 0.324360635350064 * np.ones((2, 1))
25:         # c and d in discrete should be equal to their continuous counterparts
26:         dt_requested = 0.5
27: 
28:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested, method='zoh')
29: 
30:         assert_array_almost_equal(ad_truth, ad)
31:         assert_array_almost_equal(bd_truth, bd)
32:         assert_array_almost_equal(cc, cd)
33:         assert_array_almost_equal(dc, dd)
34:         assert_almost_equal(dt_requested, dt)
35: 
36:     def test_gbt(self):
37:         ac = np.eye(2)
38:         bc = 0.5 * np.ones((2, 1))
39:         cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
40:         dc = np.array([[0.0], [0.0], [-0.33]])
41: 
42:         dt_requested = 0.5
43:         alpha = 1.0 / 3.0
44: 
45:         ad_truth = 1.6 * np.eye(2)
46:         bd_truth = 0.3 * np.ones((2, 1))
47:         cd_truth = np.array([[0.9, 1.2],
48:                              [1.2, 1.2],
49:                              [1.2, 0.3]])
50:         dd_truth = np.array([[0.175],
51:                              [0.2],
52:                              [-0.205]])
53: 
54:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
55:                                  method='gbt', alpha=alpha)
56: 
57:         assert_array_almost_equal(ad_truth, ad)
58:         assert_array_almost_equal(bd_truth, bd)
59:         assert_array_almost_equal(cd_truth, cd)
60:         assert_array_almost_equal(dd_truth, dd)
61: 
62:     def test_euler(self):
63:         ac = np.eye(2)
64:         bc = 0.5 * np.ones((2, 1))
65:         cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
66:         dc = np.array([[0.0], [0.0], [-0.33]])
67: 
68:         dt_requested = 0.5
69: 
70:         ad_truth = 1.5 * np.eye(2)
71:         bd_truth = 0.25 * np.ones((2, 1))
72:         cd_truth = np.array([[0.75, 1.0],
73:                              [1.0, 1.0],
74:                              [1.0, 0.25]])
75:         dd_truth = dc
76: 
77:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
78:                                  method='euler')
79: 
80:         assert_array_almost_equal(ad_truth, ad)
81:         assert_array_almost_equal(bd_truth, bd)
82:         assert_array_almost_equal(cd_truth, cd)
83:         assert_array_almost_equal(dd_truth, dd)
84:         assert_almost_equal(dt_requested, dt)
85: 
86:     def test_backward_diff(self):
87:         ac = np.eye(2)
88:         bc = 0.5 * np.ones((2, 1))
89:         cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
90:         dc = np.array([[0.0], [0.0], [-0.33]])
91: 
92:         dt_requested = 0.5
93: 
94:         ad_truth = 2.0 * np.eye(2)
95:         bd_truth = 0.5 * np.ones((2, 1))
96:         cd_truth = np.array([[1.5, 2.0],
97:                              [2.0, 2.0],
98:                              [2.0, 0.5]])
99:         dd_truth = np.array([[0.875],
100:                              [1.0],
101:                              [0.295]])
102: 
103:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
104:                                  method='backward_diff')
105: 
106:         assert_array_almost_equal(ad_truth, ad)
107:         assert_array_almost_equal(bd_truth, bd)
108:         assert_array_almost_equal(cd_truth, cd)
109:         assert_array_almost_equal(dd_truth, dd)
110: 
111:     def test_bilinear(self):
112:         ac = np.eye(2)
113:         bc = 0.5 * np.ones((2, 1))
114:         cc = np.array([[0.75, 1.0], [1.0, 1.0], [1.0, 0.25]])
115:         dc = np.array([[0.0], [0.0], [-0.33]])
116: 
117:         dt_requested = 0.5
118: 
119:         ad_truth = (5.0 / 3.0) * np.eye(2)
120:         bd_truth = (1.0 / 3.0) * np.ones((2, 1))
121:         cd_truth = np.array([[1.0, 4.0 / 3.0],
122:                              [4.0 / 3.0, 4.0 / 3.0],
123:                              [4.0 / 3.0, 1.0 / 3.0]])
124:         dd_truth = np.array([[0.291666666666667],
125:                              [1.0 / 3.0],
126:                              [-0.121666666666667]])
127: 
128:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
129:                                  method='bilinear')
130: 
131:         assert_array_almost_equal(ad_truth, ad)
132:         assert_array_almost_equal(bd_truth, bd)
133:         assert_array_almost_equal(cd_truth, cd)
134:         assert_array_almost_equal(dd_truth, dd)
135:         assert_almost_equal(dt_requested, dt)
136: 
137:         # Same continuous system again, but change sampling rate
138: 
139:         ad_truth = 1.4 * np.eye(2)
140:         bd_truth = 0.2 * np.ones((2, 1))
141:         cd_truth = np.array([[0.9, 1.2], [1.2, 1.2], [1.2, 0.3]])
142:         dd_truth = np.array([[0.175], [0.2], [-0.205]])
143: 
144:         dt_requested = 1.0 / 3.0
145: 
146:         ad, bd, cd, dd, dt = c2d((ac, bc, cc, dc), dt_requested,
147:                                  method='bilinear')
148: 
149:         assert_array_almost_equal(ad_truth, ad)
150:         assert_array_almost_equal(bd_truth, bd)
151:         assert_array_almost_equal(cd_truth, cd)
152:         assert_array_almost_equal(dd_truth, dd)
153:         assert_almost_equal(dt_requested, dt)
154: 
155:     def test_transferfunction(self):
156:         numc = np.array([0.25, 0.25, 0.5])
157:         denc = np.array([0.75, 0.75, 1.0])
158: 
159:         numd = np.array([[1.0 / 3.0, -0.427419169438754, 0.221654141101125]])
160:         dend = np.array([1.0, -1.351394049721225, 0.606530659712634])
161: 
162:         dt_requested = 0.5
163: 
164:         num, den, dt = c2d((numc, denc), dt_requested, method='zoh')
165: 
166:         assert_array_almost_equal(numd, num)
167:         assert_array_almost_equal(dend, den)
168:         assert_almost_equal(dt_requested, dt)
169: 
170:     def test_zerospolesgain(self):
171:         zeros_c = np.array([0.5, -0.5])
172:         poles_c = np.array([1.j / np.sqrt(2), -1.j / np.sqrt(2)])
173:         k_c = 1.0
174: 
175:         zeros_d = [1.23371727305860, 0.735356894461267]
176:         polls_d = [0.938148335039729 + 0.346233593780536j,
177:                    0.938148335039729 - 0.346233593780536j]
178:         k_d = 1.0
179: 
180:         dt_requested = 0.5
181: 
182:         zeros, poles, k, dt = c2d((zeros_c, poles_c, k_c), dt_requested,
183:                                   method='zoh')
184: 
185:         assert_array_almost_equal(zeros_d, zeros)
186:         assert_array_almost_equal(polls_d, poles)
187:         assert_almost_equal(k_d, k)
188:         assert_almost_equal(dt_requested, dt)
189: 
190:     def test_gbt_with_sio_tf_and_zpk(self):
191:         '''Test method='gbt' with alpha=0.25 for tf and zpk cases.'''
192:         # State space coefficients for the continuous SIO system.
193:         A = -1.0
194:         B = 1.0
195:         C = 1.0
196:         D = 0.5
197: 
198:         # The continuous transfer function coefficients.
199:         cnum, cden = ss2tf(A, B, C, D)
200: 
201:         # Continuous zpk representation
202:         cz, cp, ck = ss2zpk(A, B, C, D)
203: 
204:         h = 1.0
205:         alpha = 0.25
206: 
207:         # Explicit formulas, in the scalar case.
208:         Ad = (1 + (1 - alpha) * h * A) / (1 - alpha * h * A)
209:         Bd = h * B / (1 - alpha * h * A)
210:         Cd = C / (1 - alpha * h * A)
211:         Dd = D + alpha * C * Bd
212: 
213:         # Convert the explicit solution to tf
214:         dnum, dden = ss2tf(Ad, Bd, Cd, Dd)
215: 
216:         # Compute the discrete tf using cont2discrete.
217:         c2dnum, c2dden, dt = c2d((cnum, cden), h, method='gbt', alpha=alpha)
218: 
219:         assert_allclose(dnum, c2dnum)
220:         assert_allclose(dden, c2dden)
221: 
222:         # Convert explicit solution to zpk.
223:         dz, dp, dk = ss2zpk(Ad, Bd, Cd, Dd)
224: 
225:         # Compute the discrete zpk using cont2discrete.
226:         c2dz, c2dp, c2dk, dt = c2d((cz, cp, ck), h, method='gbt', alpha=alpha)
227: 
228:         assert_allclose(dz, c2dz)
229:         assert_allclose(dp, c2dp)
230:         assert_allclose(dk, c2dk)
231: 
232:     def test_discrete_approx(self):
233:         '''
234:         Test that the solution to the discrete approximation of a continuous
235:         system actually approximates the solution to the continuous system.
236:         This is an indirect test of the correctness of the implementation
237:         of cont2discrete.
238:         '''
239: 
240:         def u(t):
241:             return np.sin(2.5 * t)
242: 
243:         a = np.array([[-0.01]])
244:         b = np.array([[1.0]])
245:         c = np.array([[1.0]])
246:         d = np.array([[0.2]])
247:         x0 = 1.0
248: 
249:         t = np.linspace(0, 10.0, 101)
250:         dt = t[1] - t[0]
251:         u1 = u(t)
252: 
253:         # Use lsim2 to compute the solution to the continuous system.
254:         t, yout, xout = lsim2((a, b, c, d), T=t, U=u1, X0=x0,
255:                               rtol=1e-9, atol=1e-11)
256: 
257:         # Convert the continuous system to a discrete approximation.
258:         dsys = c2d((a, b, c, d), dt, method='bilinear')
259: 
260:         # Use dlsim with the pairwise averaged input to compute the output
261:         # of the discrete system.
262:         u2 = 0.5 * (u1[:-1] + u1[1:])
263:         t2 = t[:-1]
264:         td2, yd2, xd2 = dlsim(dsys, u=u2.reshape(-1, 1), t=t2, x0=x0)
265: 
266:         # ymid is the average of consecutive terms of the "exact" output
267:         # computed by lsim2.  This is what the discrete approximation
268:         # actually approximates.
269:         ymid = 0.5 * (yout[:-1] + yout[1:])
270: 
271:         assert_allclose(yd2.ravel(), ymid, rtol=1e-4)
272: 
273:     def test_simo_tf(self):
274:         # See gh-5753
275:         tf = ([[1, 0], [1, 1]], [1, 1])
276:         num, den, dt = c2d(tf, 0.01)
277: 
278:         assert_equal(dt, 0.01)  # sanity check
279:         assert_allclose(den, [1, -0.990404983], rtol=1e-3)
280:         assert_allclose(num, [[1, -1], [1, -0.99004983]], rtol=1e-3)
281: 
282:     def test_multioutput(self):
283:         ts = 0.01  # time step
284: 
285:         tf = ([[1, -3], [1, 5]], [1, 1])
286:         num, den, dt = c2d(tf, ts)
287: 
288:         tf1 = (tf[0][0], tf[1])
289:         num1, den1, dt1 = c2d(tf1, ts)
290: 
291:         tf2 = (tf[0][1], tf[1])
292:         num2, den2, dt2 = c2d(tf2, ts)
293: 
294:         # Sanity checks
295:         assert_equal(dt, dt1)
296:         assert_equal(dt, dt2)
297: 
298:         # Check that we get the same results
299:         assert_allclose(num, np.vstack((num1, num2)), rtol=1e-13)
300: 
301:         # Single input, so the denominator should
302:         # not be multidimensional like the numerator
303:         assert_allclose(den, den1, rtol=1e-13)
304:         assert_allclose(den, den2, rtol=1e-13)
305: 
306: class TestC2dLti(object):
307:     def test_c2d_ss(self):
308:         # StateSpace
309:         A = np.array([[-0.3, 0.1], [0.2, -0.7]])
310:         B = np.array([[0], [1]])
311:         C = np.array([[1, 0]])
312:         D = 0
313: 
314:         A_res = np.array([[0.985136404135682, 0.004876671474795],
315:                           [0.009753342949590, 0.965629718236502]])
316:         B_res = np.array([[0.000122937599964], [0.049135527547844]])
317: 
318:         sys_ssc = lti(A, B, C, D)
319:         sys_ssd = sys_ssc.to_discrete(0.05)
320: 
321:         assert_allclose(sys_ssd.A, A_res)
322:         assert_allclose(sys_ssd.B, B_res)
323:         assert_allclose(sys_ssd.C, C)
324:         assert_allclose(sys_ssd.D, D)
325: 
326:     def test_c2d_tf(self):
327: 
328:         sys = lti([0.5, 0.3], [1.0, 0.4])
329:         sys = sys.to_discrete(0.005)
330: 
331:         # Matlab results
332:         num_res = np.array([0.5, -0.485149004980066])
333:         den_res = np.array([1.0, -0.980198673306755])
334: 
335:         # Somehow a lot of numerical errors
336:         assert_allclose(sys.den, den_res, atol=0.02)
337:         assert_allclose(sys.num, num_res, atol=0.02)
338: 
339: class TestC2dLti(object):
340:     def test_c2d_ss(self):
341:         # StateSpace
342:         A = np.array([[-0.3, 0.1], [0.2, -0.7]])
343:         B = np.array([[0], [1]])
344:         C = np.array([[1, 0]])
345:         D = 0
346: 
347:         A_res = np.array([[0.985136404135682, 0.004876671474795],
348:                           [0.009753342949590, 0.965629718236502]])
349:         B_res = np.array([[0.000122937599964], [0.049135527547844]])
350: 
351:         sys_ssc = lti(A, B, C, D)
352:         sys_ssd = sys_ssc.to_discrete(0.05)
353: 
354:         assert_allclose(sys_ssd.A, A_res)
355:         assert_allclose(sys_ssd.B, B_res)
356:         assert_allclose(sys_ssd.C, C)
357:         assert_allclose(sys_ssd.D, D)
358: 
359:     def test_c2d_tf(self):
360: 
361:         sys = lti([0.5, 0.3], [1.0, 0.4])
362:         sys = sys.to_discrete(0.005)
363: 
364:         # Matlab results
365:         num_res = np.array([0.5, -0.485149004980066])
366:         den_res = np.array([1.0, -0.980198673306755])
367: 
368:         # Somehow a lot of numerical errors
369:         assert_allclose(sys.den, den_res, atol=0.02)
370:         assert_allclose(sys.num, num_res, atol=0.02)
371: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289905 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_289905) is not StypyTypeError):

    if (import_289905 != 'pyd_module'):
        __import__(import_289905)
        sys_modules_289906 = sys.modules[import_289905]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_289906.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_289905)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_allclose, assert_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289907 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_289907) is not StypyTypeError):

    if (import_289907 != 'pyd_module'):
        __import__(import_289907)
        sys_modules_289908 = sys.modules[import_289907]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_289908.module_type_store, module_type_store, ['assert_array_almost_equal', 'assert_almost_equal', 'assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_289908, sys_modules_289908.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_array_almost_equal', 'assert_almost_equal', 'assert_allclose', 'assert_equal'], [assert_array_almost_equal, assert_almost_equal, assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_289907)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import warnings' statement (line 8)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.signal import c2d' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289909 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal')

if (type(import_289909) is not StypyTypeError):

    if (import_289909 != 'pyd_module'):
        __import__(import_289909)
        sys_modules_289910 = sys.modules[import_289909]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', sys_modules_289910.module_type_store, module_type_store, ['cont2discrete'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_289910, sys_modules_289910.module_type_store, module_type_store)
    else:
        from scipy.signal import cont2discrete as c2d

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', None, module_type_store, ['cont2discrete'], [c2d])

else:
    # Assigning a type to the variable 'scipy.signal' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.signal', import_289909)

# Adding an alias
module_type_store.add_alias('c2d', 'cont2discrete')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.signal import dlsim, ss2tf, ss2zpk, lsim2, lti' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_289911 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal')

if (type(import_289911) is not StypyTypeError):

    if (import_289911 != 'pyd_module'):
        __import__(import_289911)
        sys_modules_289912 = sys.modules[import_289911]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', sys_modules_289912.module_type_store, module_type_store, ['dlsim', 'ss2tf', 'ss2zpk', 'lsim2', 'lti'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_289912, sys_modules_289912.module_type_store, module_type_store)
    else:
        from scipy.signal import dlsim, ss2tf, ss2zpk, lsim2, lti

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', None, module_type_store, ['dlsim', 'ss2tf', 'ss2zpk', 'lsim2', 'lti'], [dlsim, ss2tf, ss2zpk, lsim2, lti])

else:
    # Assigning a type to the variable 'scipy.signal' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.signal', import_289911)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

# Declaration of the 'TestC2D' class

class TestC2D(object, ):

    @norecursion
    def test_zoh(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zoh'
        module_type_store = module_type_store.open_function_context('test_zoh', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_zoh.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_zoh')
        TestC2D.test_zoh.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_zoh.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_zoh.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_zoh', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zoh', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zoh(...)' code ##################

        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to eye(...): (line 18)
        # Processing the call arguments (line 18)
        int_289915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'int')
        # Processing the call keyword arguments (line 18)
        kwargs_289916 = {}
        # Getting the type of 'np' (line 18)
        np_289913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 18)
        eye_289914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), np_289913, 'eye')
        # Calling eye(args, kwargs) (line 18)
        eye_call_result_289917 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), eye_289914, *[int_289915], **kwargs_289916)
        
        # Assigning a type to the variable 'ac' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'ac', eye_call_result_289917)
        
        # Assigning a BinOp to a Name (line 19):
        
        # Assigning a BinOp to a Name (line 19):
        float_289918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'float')
        
        # Call to ones(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_289921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        int_289922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), tuple_289921, int_289922)
        # Adding element type (line 19)
        int_289923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 28), tuple_289921, int_289923)
        
        # Processing the call keyword arguments (line 19)
        kwargs_289924 = {}
        # Getting the type of 'np' (line 19)
        np_289919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 19)
        ones_289920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), np_289919, 'ones')
        # Calling ones(args, kwargs) (line 19)
        ones_call_result_289925 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), ones_289920, *[tuple_289921], **kwargs_289924)
        
        # Applying the binary operator '*' (line 19)
        result_mul_289926 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 13), '*', float_289918, ones_call_result_289925)
        
        # Assigning a type to the variable 'bc' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'bc', result_mul_289926)
        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to array(...): (line 20)
        # Processing the call arguments (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_289929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_289930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        float_289931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_289930, float_289931)
        # Adding element type (line 20)
        float_289932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_289930, float_289932)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 22), list_289929, list_289930)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_289933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        float_289934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 36), list_289933, float_289934)
        # Adding element type (line 20)
        float_289935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 36), list_289933, float_289935)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 22), list_289929, list_289933)
        # Adding element type (line 20)
        
        # Obtaining an instance of the builtin type 'list' (line 20)
        list_289936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 20)
        # Adding element type (line 20)
        float_289937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 48), list_289936, float_289937)
        # Adding element type (line 20)
        float_289938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 48), list_289936, float_289938)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 22), list_289929, list_289936)
        
        # Processing the call keyword arguments (line 20)
        kwargs_289939 = {}
        # Getting the type of 'np' (line 20)
        np_289927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 20)
        array_289928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 13), np_289927, 'array')
        # Calling array(args, kwargs) (line 20)
        array_call_result_289940 = invoke(stypy.reporting.localization.Localization(__file__, 20, 13), array_289928, *[list_289929], **kwargs_289939)
        
        # Assigning a type to the variable 'cc' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'cc', array_call_result_289940)
        
        # Assigning a Call to a Name (line 21):
        
        # Assigning a Call to a Name (line 21):
        
        # Call to array(...): (line 21)
        # Processing the call arguments (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_289943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_289944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        float_289945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_289944, float_289945)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), list_289943, list_289944)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_289946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        float_289947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 30), list_289946, float_289947)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), list_289943, list_289946)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_289948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        float_289949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 37), list_289948, float_289949)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), list_289943, list_289948)
        
        # Processing the call keyword arguments (line 21)
        kwargs_289950 = {}
        # Getting the type of 'np' (line 21)
        np_289941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 21)
        array_289942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 13), np_289941, 'array')
        # Calling array(args, kwargs) (line 21)
        array_call_result_289951 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), array_289942, *[list_289943], **kwargs_289950)
        
        # Assigning a type to the variable 'dc' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'dc', array_call_result_289951)
        
        # Assigning a BinOp to a Name (line 23):
        
        # Assigning a BinOp to a Name (line 23):
        float_289952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'float')
        
        # Call to eye(...): (line 23)
        # Processing the call arguments (line 23)
        int_289955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'int')
        # Processing the call keyword arguments (line 23)
        kwargs_289956 = {}
        # Getting the type of 'np' (line 23)
        np_289953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'np', False)
        # Obtaining the member 'eye' of a type (line 23)
        eye_289954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 39), np_289953, 'eye')
        # Calling eye(args, kwargs) (line 23)
        eye_call_result_289957 = invoke(stypy.reporting.localization.Localization(__file__, 23, 39), eye_289954, *[int_289955], **kwargs_289956)
        
        # Applying the binary operator '*' (line 23)
        result_mul_289958 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), '*', float_289952, eye_call_result_289957)
        
        # Assigning a type to the variable 'ad_truth' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'ad_truth', result_mul_289958)
        
        # Assigning a BinOp to a Name (line 24):
        
        # Assigning a BinOp to a Name (line 24):
        float_289959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'float')
        
        # Call to ones(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_289962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        int_289963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 48), tuple_289962, int_289963)
        # Adding element type (line 24)
        int_289964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 48), tuple_289962, int_289964)
        
        # Processing the call keyword arguments (line 24)
        kwargs_289965 = {}
        # Getting the type of 'np' (line 24)
        np_289960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 39), 'np', False)
        # Obtaining the member 'ones' of a type (line 24)
        ones_289961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 39), np_289960, 'ones')
        # Calling ones(args, kwargs) (line 24)
        ones_call_result_289966 = invoke(stypy.reporting.localization.Localization(__file__, 24, 39), ones_289961, *[tuple_289962], **kwargs_289965)
        
        # Applying the binary operator '*' (line 24)
        result_mul_289967 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), '*', float_289959, ones_call_result_289966)
        
        # Assigning a type to the variable 'bd_truth' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'bd_truth', result_mul_289967)
        
        # Assigning a Num to a Name (line 26):
        
        # Assigning a Num to a Name (line 26):
        float_289968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'dt_requested', float_289968)
        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_289969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to c2d(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_289971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'ac' (line 28)
        ac_289972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289971, ac_289972)
        # Adding element type (line 28)
        # Getting the type of 'bc' (line 28)
        bc_289973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289971, bc_289973)
        # Adding element type (line 28)
        # Getting the type of 'cc' (line 28)
        cc_289974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289971, cc_289974)
        # Adding element type (line 28)
        # Getting the type of 'dc' (line 28)
        dc_289975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289971, dc_289975)
        
        # Getting the type of 'dt_requested' (line 28)
        dt_requested_289976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 28)
        str_289977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'str', 'zoh')
        keyword_289978 = str_289977
        kwargs_289979 = {'method': keyword_289978}
        # Getting the type of 'c2d' (line 28)
        c2d_289970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 28)
        c2d_call_result_289980 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), c2d_289970, *[tuple_289971, dt_requested_289976], **kwargs_289979)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___289981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), c2d_call_result_289980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_289982 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___289981, int_289969)
        
        # Assigning a type to the variable 'tuple_var_assignment_289833' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289833', subscript_call_result_289982)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_289983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to c2d(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_289985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'ac' (line 28)
        ac_289986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289985, ac_289986)
        # Adding element type (line 28)
        # Getting the type of 'bc' (line 28)
        bc_289987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289985, bc_289987)
        # Adding element type (line 28)
        # Getting the type of 'cc' (line 28)
        cc_289988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289985, cc_289988)
        # Adding element type (line 28)
        # Getting the type of 'dc' (line 28)
        dc_289989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289985, dc_289989)
        
        # Getting the type of 'dt_requested' (line 28)
        dt_requested_289990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 28)
        str_289991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'str', 'zoh')
        keyword_289992 = str_289991
        kwargs_289993 = {'method': keyword_289992}
        # Getting the type of 'c2d' (line 28)
        c2d_289984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 28)
        c2d_call_result_289994 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), c2d_289984, *[tuple_289985, dt_requested_289990], **kwargs_289993)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___289995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), c2d_call_result_289994, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_289996 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___289995, int_289983)
        
        # Assigning a type to the variable 'tuple_var_assignment_289834' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289834', subscript_call_result_289996)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_289997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to c2d(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_289999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'ac' (line 28)
        ac_290000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289999, ac_290000)
        # Adding element type (line 28)
        # Getting the type of 'bc' (line 28)
        bc_290001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289999, bc_290001)
        # Adding element type (line 28)
        # Getting the type of 'cc' (line 28)
        cc_290002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289999, cc_290002)
        # Adding element type (line 28)
        # Getting the type of 'dc' (line 28)
        dc_290003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_289999, dc_290003)
        
        # Getting the type of 'dt_requested' (line 28)
        dt_requested_290004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 28)
        str_290005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'str', 'zoh')
        keyword_290006 = str_290005
        kwargs_290007 = {'method': keyword_290006}
        # Getting the type of 'c2d' (line 28)
        c2d_289998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 28)
        c2d_call_result_290008 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), c2d_289998, *[tuple_289999, dt_requested_290004], **kwargs_290007)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___290009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), c2d_call_result_290008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_290010 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___290009, int_289997)
        
        # Assigning a type to the variable 'tuple_var_assignment_289835' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289835', subscript_call_result_290010)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_290011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to c2d(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_290013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'ac' (line 28)
        ac_290014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290013, ac_290014)
        # Adding element type (line 28)
        # Getting the type of 'bc' (line 28)
        bc_290015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290013, bc_290015)
        # Adding element type (line 28)
        # Getting the type of 'cc' (line 28)
        cc_290016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290013, cc_290016)
        # Adding element type (line 28)
        # Getting the type of 'dc' (line 28)
        dc_290017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290013, dc_290017)
        
        # Getting the type of 'dt_requested' (line 28)
        dt_requested_290018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 28)
        str_290019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'str', 'zoh')
        keyword_290020 = str_290019
        kwargs_290021 = {'method': keyword_290020}
        # Getting the type of 'c2d' (line 28)
        c2d_290012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 28)
        c2d_call_result_290022 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), c2d_290012, *[tuple_290013, dt_requested_290018], **kwargs_290021)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___290023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), c2d_call_result_290022, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_290024 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___290023, int_290011)
        
        # Assigning a type to the variable 'tuple_var_assignment_289836' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289836', subscript_call_result_290024)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_290025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to c2d(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_290027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'ac' (line 28)
        ac_290028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290027, ac_290028)
        # Adding element type (line 28)
        # Getting the type of 'bc' (line 28)
        bc_290029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290027, bc_290029)
        # Adding element type (line 28)
        # Getting the type of 'cc' (line 28)
        cc_290030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290027, cc_290030)
        # Adding element type (line 28)
        # Getting the type of 'dc' (line 28)
        dc_290031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_290027, dc_290031)
        
        # Getting the type of 'dt_requested' (line 28)
        dt_requested_290032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 28)
        str_290033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 72), 'str', 'zoh')
        keyword_290034 = str_290033
        kwargs_290035 = {'method': keyword_290034}
        # Getting the type of 'c2d' (line 28)
        c2d_290026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 28)
        c2d_call_result_290036 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), c2d_290026, *[tuple_290027, dt_requested_290032], **kwargs_290035)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___290037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), c2d_call_result_290036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_290038 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___290037, int_290025)
        
        # Assigning a type to the variable 'tuple_var_assignment_289837' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289837', subscript_call_result_290038)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_289833' (line 28)
        tuple_var_assignment_289833_290039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289833')
        # Assigning a type to the variable 'ad' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'ad', tuple_var_assignment_289833_290039)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_289834' (line 28)
        tuple_var_assignment_289834_290040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289834')
        # Assigning a type to the variable 'bd' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'bd', tuple_var_assignment_289834_290040)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_289835' (line 28)
        tuple_var_assignment_289835_290041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289835')
        # Assigning a type to the variable 'cd' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'cd', tuple_var_assignment_289835_290041)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_289836' (line 28)
        tuple_var_assignment_289836_290042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289836')
        # Assigning a type to the variable 'dd' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'dd', tuple_var_assignment_289836_290042)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_289837' (line 28)
        tuple_var_assignment_289837_290043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_289837')
        # Assigning a type to the variable 'dt' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'dt', tuple_var_assignment_289837_290043)
        
        # Call to assert_array_almost_equal(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'ad_truth' (line 30)
        ad_truth_290045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 30)
        ad_290046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 44), 'ad', False)
        # Processing the call keyword arguments (line 30)
        kwargs_290047 = {}
        # Getting the type of 'assert_array_almost_equal' (line 30)
        assert_array_almost_equal_290044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 30)
        assert_array_almost_equal_call_result_290048 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_array_almost_equal_290044, *[ad_truth_290045, ad_290046], **kwargs_290047)
        
        
        # Call to assert_array_almost_equal(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'bd_truth' (line 31)
        bd_truth_290050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 31)
        bd_290051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 44), 'bd', False)
        # Processing the call keyword arguments (line 31)
        kwargs_290052 = {}
        # Getting the type of 'assert_array_almost_equal' (line 31)
        assert_array_almost_equal_290049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 31)
        assert_array_almost_equal_call_result_290053 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), assert_array_almost_equal_290049, *[bd_truth_290050, bd_290051], **kwargs_290052)
        
        
        # Call to assert_array_almost_equal(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'cc' (line 32)
        cc_290055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 34), 'cc', False)
        # Getting the type of 'cd' (line 32)
        cd_290056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'cd', False)
        # Processing the call keyword arguments (line 32)
        kwargs_290057 = {}
        # Getting the type of 'assert_array_almost_equal' (line 32)
        assert_array_almost_equal_290054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 32)
        assert_array_almost_equal_call_result_290058 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), assert_array_almost_equal_290054, *[cc_290055, cd_290056], **kwargs_290057)
        
        
        # Call to assert_array_almost_equal(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'dc' (line 33)
        dc_290060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 34), 'dc', False)
        # Getting the type of 'dd' (line 33)
        dd_290061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'dd', False)
        # Processing the call keyword arguments (line 33)
        kwargs_290062 = {}
        # Getting the type of 'assert_array_almost_equal' (line 33)
        assert_array_almost_equal_290059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 33)
        assert_array_almost_equal_call_result_290063 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), assert_array_almost_equal_290059, *[dc_290060, dd_290061], **kwargs_290062)
        
        
        # Call to assert_almost_equal(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'dt_requested' (line 34)
        dt_requested_290065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 34)
        dt_290066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'dt', False)
        # Processing the call keyword arguments (line 34)
        kwargs_290067 = {}
        # Getting the type of 'assert_almost_equal' (line 34)
        assert_almost_equal_290064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 34)
        assert_almost_equal_call_result_290068 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), assert_almost_equal_290064, *[dt_requested_290065, dt_290066], **kwargs_290067)
        
        
        # ################# End of 'test_zoh(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zoh' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_290069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zoh'
        return stypy_return_type_290069


    @norecursion
    def test_gbt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gbt'
        module_type_store = module_type_store.open_function_context('test_gbt', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_gbt.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_gbt')
        TestC2D.test_gbt.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_gbt.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_gbt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_gbt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gbt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gbt(...)' code ##################

        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to eye(...): (line 37)
        # Processing the call arguments (line 37)
        int_290072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'int')
        # Processing the call keyword arguments (line 37)
        kwargs_290073 = {}
        # Getting the type of 'np' (line 37)
        np_290070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 37)
        eye_290071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 13), np_290070, 'eye')
        # Calling eye(args, kwargs) (line 37)
        eye_call_result_290074 = invoke(stypy.reporting.localization.Localization(__file__, 37, 13), eye_290071, *[int_290072], **kwargs_290073)
        
        # Assigning a type to the variable 'ac' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'ac', eye_call_result_290074)
        
        # Assigning a BinOp to a Name (line 38):
        
        # Assigning a BinOp to a Name (line 38):
        float_290075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'float')
        
        # Call to ones(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_290078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        int_290079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 28), tuple_290078, int_290079)
        # Adding element type (line 38)
        int_290080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 28), tuple_290078, int_290080)
        
        # Processing the call keyword arguments (line 38)
        kwargs_290081 = {}
        # Getting the type of 'np' (line 38)
        np_290076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 38)
        ones_290077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), np_290076, 'ones')
        # Calling ones(args, kwargs) (line 38)
        ones_call_result_290082 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), ones_290077, *[tuple_290078], **kwargs_290081)
        
        # Applying the binary operator '*' (line 38)
        result_mul_290083 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 13), '*', float_290075, ones_call_result_290082)
        
        # Assigning a type to the variable 'bc' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'bc', result_mul_290083)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to array(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_290086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_290087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        float_290088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_290087, float_290088)
        # Adding element type (line 39)
        float_290089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 23), list_290087, float_290089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_290086, list_290087)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_290090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        float_290091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), list_290090, float_290091)
        # Adding element type (line 39)
        float_290092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), list_290090, float_290092)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_290086, list_290090)
        # Adding element type (line 39)
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_290093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        # Adding element type (line 39)
        float_290094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), list_290093, float_290094)
        # Adding element type (line 39)
        float_290095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 48), list_290093, float_290095)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 22), list_290086, list_290093)
        
        # Processing the call keyword arguments (line 39)
        kwargs_290096 = {}
        # Getting the type of 'np' (line 39)
        np_290084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 39)
        array_290085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), np_290084, 'array')
        # Calling array(args, kwargs) (line 39)
        array_call_result_290097 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), array_290085, *[list_290086], **kwargs_290096)
        
        # Assigning a type to the variable 'cc' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'cc', array_call_result_290097)
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to array(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_290100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_290101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        float_290102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 23), list_290101, float_290102)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 22), list_290100, list_290101)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_290103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        float_290104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 30), list_290103, float_290104)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 22), list_290100, list_290103)
        # Adding element type (line 40)
        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_290105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        float_290106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 37), list_290105, float_290106)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 22), list_290100, list_290105)
        
        # Processing the call keyword arguments (line 40)
        kwargs_290107 = {}
        # Getting the type of 'np' (line 40)
        np_290098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 40)
        array_290099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 13), np_290098, 'array')
        # Calling array(args, kwargs) (line 40)
        array_call_result_290108 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), array_290099, *[list_290100], **kwargs_290107)
        
        # Assigning a type to the variable 'dc' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'dc', array_call_result_290108)
        
        # Assigning a Num to a Name (line 42):
        
        # Assigning a Num to a Name (line 42):
        float_290109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'dt_requested', float_290109)
        
        # Assigning a BinOp to a Name (line 43):
        
        # Assigning a BinOp to a Name (line 43):
        float_290110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'float')
        float_290111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'float')
        # Applying the binary operator 'div' (line 43)
        result_div_290112 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 16), 'div', float_290110, float_290111)
        
        # Assigning a type to the variable 'alpha' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'alpha', result_div_290112)
        
        # Assigning a BinOp to a Name (line 45):
        
        # Assigning a BinOp to a Name (line 45):
        float_290113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 19), 'float')
        
        # Call to eye(...): (line 45)
        # Processing the call arguments (line 45)
        int_290116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 32), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_290117 = {}
        # Getting the type of 'np' (line 45)
        np_290114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 25), 'np', False)
        # Obtaining the member 'eye' of a type (line 45)
        eye_290115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 25), np_290114, 'eye')
        # Calling eye(args, kwargs) (line 45)
        eye_call_result_290118 = invoke(stypy.reporting.localization.Localization(__file__, 45, 25), eye_290115, *[int_290116], **kwargs_290117)
        
        # Applying the binary operator '*' (line 45)
        result_mul_290119 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), '*', float_290113, eye_call_result_290118)
        
        # Assigning a type to the variable 'ad_truth' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'ad_truth', result_mul_290119)
        
        # Assigning a BinOp to a Name (line 46):
        
        # Assigning a BinOp to a Name (line 46):
        float_290120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 19), 'float')
        
        # Call to ones(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_290123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        int_290124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_290123, int_290124)
        # Adding element type (line 46)
        int_290125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 34), tuple_290123, int_290125)
        
        # Processing the call keyword arguments (line 46)
        kwargs_290126 = {}
        # Getting the type of 'np' (line 46)
        np_290121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'np', False)
        # Obtaining the member 'ones' of a type (line 46)
        ones_290122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), np_290121, 'ones')
        # Calling ones(args, kwargs) (line 46)
        ones_call_result_290127 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), ones_290122, *[tuple_290123], **kwargs_290126)
        
        # Applying the binary operator '*' (line 46)
        result_mul_290128 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 19), '*', float_290120, ones_call_result_290127)
        
        # Assigning a type to the variable 'bd_truth' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'bd_truth', result_mul_290128)
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to array(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_290131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_290132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        float_290133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), list_290132, float_290133)
        # Adding element type (line 47)
        float_290134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 29), list_290132, float_290134)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_290131, list_290132)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_290135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        float_290136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 29), list_290135, float_290136)
        # Adding element type (line 48)
        float_290137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 29), list_290135, float_290137)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_290131, list_290135)
        # Adding element type (line 47)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_290138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        float_290139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_290138, float_290139)
        # Adding element type (line 49)
        float_290140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_290138, float_290140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 28), list_290131, list_290138)
        
        # Processing the call keyword arguments (line 47)
        kwargs_290141 = {}
        # Getting the type of 'np' (line 47)
        np_290129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 47)
        array_290130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), np_290129, 'array')
        # Calling array(args, kwargs) (line 47)
        array_call_result_290142 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), array_290130, *[list_290131], **kwargs_290141)
        
        # Assigning a type to the variable 'cd_truth' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'cd_truth', array_call_result_290142)
        
        # Assigning a Call to a Name (line 50):
        
        # Assigning a Call to a Name (line 50):
        
        # Call to array(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_290145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        
        # Obtaining an instance of the builtin type 'list' (line 50)
        list_290146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 50)
        # Adding element type (line 50)
        float_290147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 29), list_290146, float_290147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 28), list_290145, list_290146)
        # Adding element type (line 50)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_290148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        float_290149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 29), list_290148, float_290149)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 28), list_290145, list_290148)
        # Adding element type (line 50)
        
        # Obtaining an instance of the builtin type 'list' (line 52)
        list_290150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 52)
        # Adding element type (line 52)
        float_290151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 29), list_290150, float_290151)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 28), list_290145, list_290150)
        
        # Processing the call keyword arguments (line 50)
        kwargs_290152 = {}
        # Getting the type of 'np' (line 50)
        np_290143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 50)
        array_290144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), np_290143, 'array')
        # Calling array(args, kwargs) (line 50)
        array_call_result_290153 = invoke(stypy.reporting.localization.Localization(__file__, 50, 19), array_290144, *[list_290145], **kwargs_290152)
        
        # Assigning a type to the variable 'dd_truth' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'dd_truth', array_call_result_290153)
        
        # Assigning a Call to a Tuple (line 54):
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_290154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to c2d(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_290156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ac' (line 54)
        ac_290157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290156, ac_290157)
        # Adding element type (line 54)
        # Getting the type of 'bc' (line 54)
        bc_290158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290156, bc_290158)
        # Adding element type (line 54)
        # Getting the type of 'cc' (line 54)
        cc_290159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290156, cc_290159)
        # Adding element type (line 54)
        # Getting the type of 'dc' (line 54)
        dc_290160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290156, dc_290160)
        
        # Getting the type of 'dt_requested' (line 54)
        dt_requested_290161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 54)
        str_290162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'gbt')
        keyword_290163 = str_290162
        # Getting the type of 'alpha' (line 55)
        alpha_290164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'alpha', False)
        keyword_290165 = alpha_290164
        kwargs_290166 = {'alpha': keyword_290165, 'method': keyword_290163}
        # Getting the type of 'c2d' (line 54)
        c2d_290155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 54)
        c2d_call_result_290167 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), c2d_290155, *[tuple_290156, dt_requested_290161], **kwargs_290166)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___290168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), c2d_call_result_290167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_290169 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___290168, int_290154)
        
        # Assigning a type to the variable 'tuple_var_assignment_289838' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289838', subscript_call_result_290169)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_290170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to c2d(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_290172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ac' (line 54)
        ac_290173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290172, ac_290173)
        # Adding element type (line 54)
        # Getting the type of 'bc' (line 54)
        bc_290174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290172, bc_290174)
        # Adding element type (line 54)
        # Getting the type of 'cc' (line 54)
        cc_290175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290172, cc_290175)
        # Adding element type (line 54)
        # Getting the type of 'dc' (line 54)
        dc_290176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290172, dc_290176)
        
        # Getting the type of 'dt_requested' (line 54)
        dt_requested_290177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 54)
        str_290178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'gbt')
        keyword_290179 = str_290178
        # Getting the type of 'alpha' (line 55)
        alpha_290180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'alpha', False)
        keyword_290181 = alpha_290180
        kwargs_290182 = {'alpha': keyword_290181, 'method': keyword_290179}
        # Getting the type of 'c2d' (line 54)
        c2d_290171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 54)
        c2d_call_result_290183 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), c2d_290171, *[tuple_290172, dt_requested_290177], **kwargs_290182)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___290184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), c2d_call_result_290183, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_290185 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___290184, int_290170)
        
        # Assigning a type to the variable 'tuple_var_assignment_289839' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289839', subscript_call_result_290185)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_290186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to c2d(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_290188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ac' (line 54)
        ac_290189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290188, ac_290189)
        # Adding element type (line 54)
        # Getting the type of 'bc' (line 54)
        bc_290190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290188, bc_290190)
        # Adding element type (line 54)
        # Getting the type of 'cc' (line 54)
        cc_290191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290188, cc_290191)
        # Adding element type (line 54)
        # Getting the type of 'dc' (line 54)
        dc_290192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290188, dc_290192)
        
        # Getting the type of 'dt_requested' (line 54)
        dt_requested_290193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 54)
        str_290194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'gbt')
        keyword_290195 = str_290194
        # Getting the type of 'alpha' (line 55)
        alpha_290196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'alpha', False)
        keyword_290197 = alpha_290196
        kwargs_290198 = {'alpha': keyword_290197, 'method': keyword_290195}
        # Getting the type of 'c2d' (line 54)
        c2d_290187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 54)
        c2d_call_result_290199 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), c2d_290187, *[tuple_290188, dt_requested_290193], **kwargs_290198)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___290200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), c2d_call_result_290199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_290201 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___290200, int_290186)
        
        # Assigning a type to the variable 'tuple_var_assignment_289840' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289840', subscript_call_result_290201)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_290202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to c2d(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_290204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ac' (line 54)
        ac_290205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290204, ac_290205)
        # Adding element type (line 54)
        # Getting the type of 'bc' (line 54)
        bc_290206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290204, bc_290206)
        # Adding element type (line 54)
        # Getting the type of 'cc' (line 54)
        cc_290207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290204, cc_290207)
        # Adding element type (line 54)
        # Getting the type of 'dc' (line 54)
        dc_290208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290204, dc_290208)
        
        # Getting the type of 'dt_requested' (line 54)
        dt_requested_290209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 54)
        str_290210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'gbt')
        keyword_290211 = str_290210
        # Getting the type of 'alpha' (line 55)
        alpha_290212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'alpha', False)
        keyword_290213 = alpha_290212
        kwargs_290214 = {'alpha': keyword_290213, 'method': keyword_290211}
        # Getting the type of 'c2d' (line 54)
        c2d_290203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 54)
        c2d_call_result_290215 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), c2d_290203, *[tuple_290204, dt_requested_290209], **kwargs_290214)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___290216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), c2d_call_result_290215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_290217 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___290216, int_290202)
        
        # Assigning a type to the variable 'tuple_var_assignment_289841' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289841', subscript_call_result_290217)
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_290218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'int')
        
        # Call to c2d(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_290220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'ac' (line 54)
        ac_290221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290220, ac_290221)
        # Adding element type (line 54)
        # Getting the type of 'bc' (line 54)
        bc_290222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290220, bc_290222)
        # Adding element type (line 54)
        # Getting the type of 'cc' (line 54)
        cc_290223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290220, cc_290223)
        # Adding element type (line 54)
        # Getting the type of 'dc' (line 54)
        dc_290224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 34), tuple_290220, dc_290224)
        
        # Getting the type of 'dt_requested' (line 54)
        dt_requested_290225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 54)
        str_290226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'str', 'gbt')
        keyword_290227 = str_290226
        # Getting the type of 'alpha' (line 55)
        alpha_290228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 53), 'alpha', False)
        keyword_290229 = alpha_290228
        kwargs_290230 = {'alpha': keyword_290229, 'method': keyword_290227}
        # Getting the type of 'c2d' (line 54)
        c2d_290219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 54)
        c2d_call_result_290231 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), c2d_290219, *[tuple_290220, dt_requested_290225], **kwargs_290230)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___290232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), c2d_call_result_290231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_290233 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), getitem___290232, int_290218)
        
        # Assigning a type to the variable 'tuple_var_assignment_289842' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289842', subscript_call_result_290233)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_289838' (line 54)
        tuple_var_assignment_289838_290234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289838')
        # Assigning a type to the variable 'ad' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'ad', tuple_var_assignment_289838_290234)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_289839' (line 54)
        tuple_var_assignment_289839_290235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289839')
        # Assigning a type to the variable 'bd' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'bd', tuple_var_assignment_289839_290235)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_289840' (line 54)
        tuple_var_assignment_289840_290236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289840')
        # Assigning a type to the variable 'cd' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'cd', tuple_var_assignment_289840_290236)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_289841' (line 54)
        tuple_var_assignment_289841_290237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289841')
        # Assigning a type to the variable 'dd' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'dd', tuple_var_assignment_289841_290237)
        
        # Assigning a Name to a Name (line 54):
        # Getting the type of 'tuple_var_assignment_289842' (line 54)
        tuple_var_assignment_289842_290238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tuple_var_assignment_289842')
        # Assigning a type to the variable 'dt' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'dt', tuple_var_assignment_289842_290238)
        
        # Call to assert_array_almost_equal(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'ad_truth' (line 57)
        ad_truth_290240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 57)
        ad_290241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'ad', False)
        # Processing the call keyword arguments (line 57)
        kwargs_290242 = {}
        # Getting the type of 'assert_array_almost_equal' (line 57)
        assert_array_almost_equal_290239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 57)
        assert_array_almost_equal_call_result_290243 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), assert_array_almost_equal_290239, *[ad_truth_290240, ad_290241], **kwargs_290242)
        
        
        # Call to assert_array_almost_equal(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'bd_truth' (line 58)
        bd_truth_290245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 58)
        bd_290246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 44), 'bd', False)
        # Processing the call keyword arguments (line 58)
        kwargs_290247 = {}
        # Getting the type of 'assert_array_almost_equal' (line 58)
        assert_array_almost_equal_290244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 58)
        assert_array_almost_equal_call_result_290248 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert_array_almost_equal_290244, *[bd_truth_290245, bd_290246], **kwargs_290247)
        
        
        # Call to assert_array_almost_equal(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'cd_truth' (line 59)
        cd_truth_290250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'cd_truth', False)
        # Getting the type of 'cd' (line 59)
        cd_290251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 44), 'cd', False)
        # Processing the call keyword arguments (line 59)
        kwargs_290252 = {}
        # Getting the type of 'assert_array_almost_equal' (line 59)
        assert_array_almost_equal_290249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 59)
        assert_array_almost_equal_call_result_290253 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), assert_array_almost_equal_290249, *[cd_truth_290250, cd_290251], **kwargs_290252)
        
        
        # Call to assert_array_almost_equal(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'dd_truth' (line 60)
        dd_truth_290255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'dd_truth', False)
        # Getting the type of 'dd' (line 60)
        dd_290256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'dd', False)
        # Processing the call keyword arguments (line 60)
        kwargs_290257 = {}
        # Getting the type of 'assert_array_almost_equal' (line 60)
        assert_array_almost_equal_290254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 60)
        assert_array_almost_equal_call_result_290258 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_array_almost_equal_290254, *[dd_truth_290255, dd_290256], **kwargs_290257)
        
        
        # ################# End of 'test_gbt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gbt' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_290259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gbt'
        return stypy_return_type_290259


    @norecursion
    def test_euler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_euler'
        module_type_store = module_type_store.open_function_context('test_euler', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_euler.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_euler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_euler.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_euler.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_euler')
        TestC2D.test_euler.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_euler.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_euler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_euler.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_euler.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_euler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_euler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_euler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_euler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_euler(...)' code ##################

        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to eye(...): (line 63)
        # Processing the call arguments (line 63)
        int_290262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'int')
        # Processing the call keyword arguments (line 63)
        kwargs_290263 = {}
        # Getting the type of 'np' (line 63)
        np_290260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 63)
        eye_290261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 13), np_290260, 'eye')
        # Calling eye(args, kwargs) (line 63)
        eye_call_result_290264 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), eye_290261, *[int_290262], **kwargs_290263)
        
        # Assigning a type to the variable 'ac' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'ac', eye_call_result_290264)
        
        # Assigning a BinOp to a Name (line 64):
        
        # Assigning a BinOp to a Name (line 64):
        float_290265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 13), 'float')
        
        # Call to ones(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Obtaining an instance of the builtin type 'tuple' (line 64)
        tuple_290268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 64)
        # Adding element type (line 64)
        int_290269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 28), tuple_290268, int_290269)
        # Adding element type (line 64)
        int_290270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 28), tuple_290268, int_290270)
        
        # Processing the call keyword arguments (line 64)
        kwargs_290271 = {}
        # Getting the type of 'np' (line 64)
        np_290266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 64)
        ones_290267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), np_290266, 'ones')
        # Calling ones(args, kwargs) (line 64)
        ones_call_result_290272 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), ones_290267, *[tuple_290268], **kwargs_290271)
        
        # Applying the binary operator '*' (line 64)
        result_mul_290273 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 13), '*', float_290265, ones_call_result_290272)
        
        # Assigning a type to the variable 'bc' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'bc', result_mul_290273)
        
        # Assigning a Call to a Name (line 65):
        
        # Assigning a Call to a Name (line 65):
        
        # Call to array(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_290276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_290277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_290278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 23), list_290277, float_290278)
        # Adding element type (line 65)
        float_290279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 23), list_290277, float_290279)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_290276, list_290277)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_290280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_290281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 36), list_290280, float_290281)
        # Adding element type (line 65)
        float_290282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 36), list_290280, float_290282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_290276, list_290280)
        # Adding element type (line 65)
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_290283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        float_290284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 48), list_290283, float_290284)
        # Adding element type (line 65)
        float_290285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 48), list_290283, float_290285)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 22), list_290276, list_290283)
        
        # Processing the call keyword arguments (line 65)
        kwargs_290286 = {}
        # Getting the type of 'np' (line 65)
        np_290274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 65)
        array_290275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), np_290274, 'array')
        # Calling array(args, kwargs) (line 65)
        array_call_result_290287 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), array_290275, *[list_290276], **kwargs_290286)
        
        # Assigning a type to the variable 'cc' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'cc', array_call_result_290287)
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to array(...): (line 66)
        # Processing the call arguments (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_290290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_290291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_290292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 23), list_290291, float_290292)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_290290, list_290291)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_290293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_290294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 30), list_290293, float_290294)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_290290, list_290293)
        # Adding element type (line 66)
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_290295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        # Adding element type (line 66)
        float_290296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 37), list_290295, float_290296)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 22), list_290290, list_290295)
        
        # Processing the call keyword arguments (line 66)
        kwargs_290297 = {}
        # Getting the type of 'np' (line 66)
        np_290288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 66)
        array_290289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 13), np_290288, 'array')
        # Calling array(args, kwargs) (line 66)
        array_call_result_290298 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), array_290289, *[list_290290], **kwargs_290297)
        
        # Assigning a type to the variable 'dc' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'dc', array_call_result_290298)
        
        # Assigning a Num to a Name (line 68):
        
        # Assigning a Num to a Name (line 68):
        float_290299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'dt_requested', float_290299)
        
        # Assigning a BinOp to a Name (line 70):
        
        # Assigning a BinOp to a Name (line 70):
        float_290300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'float')
        
        # Call to eye(...): (line 70)
        # Processing the call arguments (line 70)
        int_290303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 32), 'int')
        # Processing the call keyword arguments (line 70)
        kwargs_290304 = {}
        # Getting the type of 'np' (line 70)
        np_290301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'np', False)
        # Obtaining the member 'eye' of a type (line 70)
        eye_290302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 25), np_290301, 'eye')
        # Calling eye(args, kwargs) (line 70)
        eye_call_result_290305 = invoke(stypy.reporting.localization.Localization(__file__, 70, 25), eye_290302, *[int_290303], **kwargs_290304)
        
        # Applying the binary operator '*' (line 70)
        result_mul_290306 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '*', float_290300, eye_call_result_290305)
        
        # Assigning a type to the variable 'ad_truth' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'ad_truth', result_mul_290306)
        
        # Assigning a BinOp to a Name (line 71):
        
        # Assigning a BinOp to a Name (line 71):
        float_290307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'float')
        
        # Call to ones(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_290310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        int_290311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 35), tuple_290310, int_290311)
        # Adding element type (line 71)
        int_290312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 35), tuple_290310, int_290312)
        
        # Processing the call keyword arguments (line 71)
        kwargs_290313 = {}
        # Getting the type of 'np' (line 71)
        np_290308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'np', False)
        # Obtaining the member 'ones' of a type (line 71)
        ones_290309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), np_290308, 'ones')
        # Calling ones(args, kwargs) (line 71)
        ones_call_result_290314 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), ones_290309, *[tuple_290310], **kwargs_290313)
        
        # Applying the binary operator '*' (line 71)
        result_mul_290315 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), '*', float_290307, ones_call_result_290314)
        
        # Assigning a type to the variable 'bd_truth' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'bd_truth', result_mul_290315)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to array(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_290318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_290319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        float_290320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 29), list_290319, float_290320)
        # Adding element type (line 72)
        float_290321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 29), list_290319, float_290321)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), list_290318, list_290319)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_290322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        float_290323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_290322, float_290323)
        # Adding element type (line 73)
        float_290324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 29), list_290322, float_290324)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), list_290318, list_290322)
        # Adding element type (line 72)
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_290325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        # Adding element type (line 74)
        float_290326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_290325, float_290326)
        # Adding element type (line 74)
        float_290327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 29), list_290325, float_290327)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 28), list_290318, list_290325)
        
        # Processing the call keyword arguments (line 72)
        kwargs_290328 = {}
        # Getting the type of 'np' (line 72)
        np_290316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 72)
        array_290317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), np_290316, 'array')
        # Calling array(args, kwargs) (line 72)
        array_call_result_290329 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), array_290317, *[list_290318], **kwargs_290328)
        
        # Assigning a type to the variable 'cd_truth' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'cd_truth', array_call_result_290329)
        
        # Assigning a Name to a Name (line 75):
        
        # Assigning a Name to a Name (line 75):
        # Getting the type of 'dc' (line 75)
        dc_290330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'dc')
        # Assigning a type to the variable 'dd_truth' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'dd_truth', dc_290330)
        
        # Assigning a Call to a Tuple (line 77):
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_290331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
        
        # Call to c2d(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_290333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ac' (line 77)
        ac_290334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290333, ac_290334)
        # Adding element type (line 77)
        # Getting the type of 'bc' (line 77)
        bc_290335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290333, bc_290335)
        # Adding element type (line 77)
        # Getting the type of 'cc' (line 77)
        cc_290336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290333, cc_290336)
        # Adding element type (line 77)
        # Getting the type of 'dc' (line 77)
        dc_290337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290333, dc_290337)
        
        # Getting the type of 'dt_requested' (line 77)
        dt_requested_290338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 77)
        str_290339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'str', 'euler')
        keyword_290340 = str_290339
        kwargs_290341 = {'method': keyword_290340}
        # Getting the type of 'c2d' (line 77)
        c2d_290332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 77)
        c2d_call_result_290342 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), c2d_290332, *[tuple_290333, dt_requested_290338], **kwargs_290341)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___290343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), c2d_call_result_290342, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_290344 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___290343, int_290331)
        
        # Assigning a type to the variable 'tuple_var_assignment_289843' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289843', subscript_call_result_290344)
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_290345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
        
        # Call to c2d(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_290347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ac' (line 77)
        ac_290348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290347, ac_290348)
        # Adding element type (line 77)
        # Getting the type of 'bc' (line 77)
        bc_290349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290347, bc_290349)
        # Adding element type (line 77)
        # Getting the type of 'cc' (line 77)
        cc_290350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290347, cc_290350)
        # Adding element type (line 77)
        # Getting the type of 'dc' (line 77)
        dc_290351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290347, dc_290351)
        
        # Getting the type of 'dt_requested' (line 77)
        dt_requested_290352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 77)
        str_290353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'str', 'euler')
        keyword_290354 = str_290353
        kwargs_290355 = {'method': keyword_290354}
        # Getting the type of 'c2d' (line 77)
        c2d_290346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 77)
        c2d_call_result_290356 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), c2d_290346, *[tuple_290347, dt_requested_290352], **kwargs_290355)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___290357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), c2d_call_result_290356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_290358 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___290357, int_290345)
        
        # Assigning a type to the variable 'tuple_var_assignment_289844' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289844', subscript_call_result_290358)
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_290359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
        
        # Call to c2d(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_290361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ac' (line 77)
        ac_290362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290361, ac_290362)
        # Adding element type (line 77)
        # Getting the type of 'bc' (line 77)
        bc_290363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290361, bc_290363)
        # Adding element type (line 77)
        # Getting the type of 'cc' (line 77)
        cc_290364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290361, cc_290364)
        # Adding element type (line 77)
        # Getting the type of 'dc' (line 77)
        dc_290365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290361, dc_290365)
        
        # Getting the type of 'dt_requested' (line 77)
        dt_requested_290366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 77)
        str_290367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'str', 'euler')
        keyword_290368 = str_290367
        kwargs_290369 = {'method': keyword_290368}
        # Getting the type of 'c2d' (line 77)
        c2d_290360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 77)
        c2d_call_result_290370 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), c2d_290360, *[tuple_290361, dt_requested_290366], **kwargs_290369)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___290371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), c2d_call_result_290370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_290372 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___290371, int_290359)
        
        # Assigning a type to the variable 'tuple_var_assignment_289845' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289845', subscript_call_result_290372)
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_290373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
        
        # Call to c2d(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_290375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ac' (line 77)
        ac_290376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290375, ac_290376)
        # Adding element type (line 77)
        # Getting the type of 'bc' (line 77)
        bc_290377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290375, bc_290377)
        # Adding element type (line 77)
        # Getting the type of 'cc' (line 77)
        cc_290378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290375, cc_290378)
        # Adding element type (line 77)
        # Getting the type of 'dc' (line 77)
        dc_290379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290375, dc_290379)
        
        # Getting the type of 'dt_requested' (line 77)
        dt_requested_290380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 77)
        str_290381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'str', 'euler')
        keyword_290382 = str_290381
        kwargs_290383 = {'method': keyword_290382}
        # Getting the type of 'c2d' (line 77)
        c2d_290374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 77)
        c2d_call_result_290384 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), c2d_290374, *[tuple_290375, dt_requested_290380], **kwargs_290383)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___290385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), c2d_call_result_290384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_290386 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___290385, int_290373)
        
        # Assigning a type to the variable 'tuple_var_assignment_289846' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289846', subscript_call_result_290386)
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_290387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'int')
        
        # Call to c2d(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_290389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        # Getting the type of 'ac' (line 77)
        ac_290390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290389, ac_290390)
        # Adding element type (line 77)
        # Getting the type of 'bc' (line 77)
        bc_290391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290389, bc_290391)
        # Adding element type (line 77)
        # Getting the type of 'cc' (line 77)
        cc_290392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290389, cc_290392)
        # Adding element type (line 77)
        # Getting the type of 'dc' (line 77)
        dc_290393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_290389, dc_290393)
        
        # Getting the type of 'dt_requested' (line 77)
        dt_requested_290394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 77)
        str_290395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'str', 'euler')
        keyword_290396 = str_290395
        kwargs_290397 = {'method': keyword_290396}
        # Getting the type of 'c2d' (line 77)
        c2d_290388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 77)
        c2d_call_result_290398 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), c2d_290388, *[tuple_290389, dt_requested_290394], **kwargs_290397)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___290399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), c2d_call_result_290398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_290400 = invoke(stypy.reporting.localization.Localization(__file__, 77, 8), getitem___290399, int_290387)
        
        # Assigning a type to the variable 'tuple_var_assignment_289847' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289847', subscript_call_result_290400)
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'tuple_var_assignment_289843' (line 77)
        tuple_var_assignment_289843_290401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289843')
        # Assigning a type to the variable 'ad' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'ad', tuple_var_assignment_289843_290401)
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'tuple_var_assignment_289844' (line 77)
        tuple_var_assignment_289844_290402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289844')
        # Assigning a type to the variable 'bd' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'bd', tuple_var_assignment_289844_290402)
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'tuple_var_assignment_289845' (line 77)
        tuple_var_assignment_289845_290403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289845')
        # Assigning a type to the variable 'cd' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'cd', tuple_var_assignment_289845_290403)
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'tuple_var_assignment_289846' (line 77)
        tuple_var_assignment_289846_290404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289846')
        # Assigning a type to the variable 'dd' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'dd', tuple_var_assignment_289846_290404)
        
        # Assigning a Name to a Name (line 77):
        # Getting the type of 'tuple_var_assignment_289847' (line 77)
        tuple_var_assignment_289847_290405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'tuple_var_assignment_289847')
        # Assigning a type to the variable 'dt' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'dt', tuple_var_assignment_289847_290405)
        
        # Call to assert_array_almost_equal(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'ad_truth' (line 80)
        ad_truth_290407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 80)
        ad_290408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'ad', False)
        # Processing the call keyword arguments (line 80)
        kwargs_290409 = {}
        # Getting the type of 'assert_array_almost_equal' (line 80)
        assert_array_almost_equal_290406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 80)
        assert_array_almost_equal_call_result_290410 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_array_almost_equal_290406, *[ad_truth_290407, ad_290408], **kwargs_290409)
        
        
        # Call to assert_array_almost_equal(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'bd_truth' (line 81)
        bd_truth_290412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 81)
        bd_290413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 44), 'bd', False)
        # Processing the call keyword arguments (line 81)
        kwargs_290414 = {}
        # Getting the type of 'assert_array_almost_equal' (line 81)
        assert_array_almost_equal_290411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 81)
        assert_array_almost_equal_call_result_290415 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), assert_array_almost_equal_290411, *[bd_truth_290412, bd_290413], **kwargs_290414)
        
        
        # Call to assert_array_almost_equal(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'cd_truth' (line 82)
        cd_truth_290417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'cd_truth', False)
        # Getting the type of 'cd' (line 82)
        cd_290418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 44), 'cd', False)
        # Processing the call keyword arguments (line 82)
        kwargs_290419 = {}
        # Getting the type of 'assert_array_almost_equal' (line 82)
        assert_array_almost_equal_290416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 82)
        assert_array_almost_equal_call_result_290420 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert_array_almost_equal_290416, *[cd_truth_290417, cd_290418], **kwargs_290419)
        
        
        # Call to assert_array_almost_equal(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'dd_truth' (line 83)
        dd_truth_290422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'dd_truth', False)
        # Getting the type of 'dd' (line 83)
        dd_290423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 44), 'dd', False)
        # Processing the call keyword arguments (line 83)
        kwargs_290424 = {}
        # Getting the type of 'assert_array_almost_equal' (line 83)
        assert_array_almost_equal_290421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 83)
        assert_array_almost_equal_call_result_290425 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), assert_array_almost_equal_290421, *[dd_truth_290422, dd_290423], **kwargs_290424)
        
        
        # Call to assert_almost_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'dt_requested' (line 84)
        dt_requested_290427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 84)
        dt_290428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'dt', False)
        # Processing the call keyword arguments (line 84)
        kwargs_290429 = {}
        # Getting the type of 'assert_almost_equal' (line 84)
        assert_almost_equal_290426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 84)
        assert_almost_equal_call_result_290430 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_almost_equal_290426, *[dt_requested_290427, dt_290428], **kwargs_290429)
        
        
        # ################# End of 'test_euler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_euler' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_290431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_euler'
        return stypy_return_type_290431


    @norecursion
    def test_backward_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_backward_diff'
        module_type_store = module_type_store.open_function_context('test_backward_diff', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_backward_diff')
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_backward_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_backward_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_backward_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_backward_diff(...)' code ##################

        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to eye(...): (line 87)
        # Processing the call arguments (line 87)
        int_290434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_290435 = {}
        # Getting the type of 'np' (line 87)
        np_290432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 87)
        eye_290433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 13), np_290432, 'eye')
        # Calling eye(args, kwargs) (line 87)
        eye_call_result_290436 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), eye_290433, *[int_290434], **kwargs_290435)
        
        # Assigning a type to the variable 'ac' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ac', eye_call_result_290436)
        
        # Assigning a BinOp to a Name (line 88):
        
        # Assigning a BinOp to a Name (line 88):
        float_290437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'float')
        
        # Call to ones(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_290440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        int_290441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), tuple_290440, int_290441)
        # Adding element type (line 88)
        int_290442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), tuple_290440, int_290442)
        
        # Processing the call keyword arguments (line 88)
        kwargs_290443 = {}
        # Getting the type of 'np' (line 88)
        np_290438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 88)
        ones_290439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), np_290438, 'ones')
        # Calling ones(args, kwargs) (line 88)
        ones_call_result_290444 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), ones_290439, *[tuple_290440], **kwargs_290443)
        
        # Applying the binary operator '*' (line 88)
        result_mul_290445 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 13), '*', float_290437, ones_call_result_290444)
        
        # Assigning a type to the variable 'bc' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'bc', result_mul_290445)
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to array(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_290448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_290449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        float_290450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 23), list_290449, float_290450)
        # Adding element type (line 89)
        float_290451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 23), list_290449, float_290451)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), list_290448, list_290449)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_290452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        float_290453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), list_290452, float_290453)
        # Adding element type (line 89)
        float_290454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 36), list_290452, float_290454)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), list_290448, list_290452)
        # Adding element type (line 89)
        
        # Obtaining an instance of the builtin type 'list' (line 89)
        list_290455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 89)
        # Adding element type (line 89)
        float_290456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 48), list_290455, float_290456)
        # Adding element type (line 89)
        float_290457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 48), list_290455, float_290457)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), list_290448, list_290455)
        
        # Processing the call keyword arguments (line 89)
        kwargs_290458 = {}
        # Getting the type of 'np' (line 89)
        np_290446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 89)
        array_290447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), np_290446, 'array')
        # Calling array(args, kwargs) (line 89)
        array_call_result_290459 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), array_290447, *[list_290448], **kwargs_290458)
        
        # Assigning a type to the variable 'cc' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'cc', array_call_result_290459)
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to array(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_290462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_290463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_290464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 23), list_290463, float_290464)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 22), list_290462, list_290463)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_290465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_290466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 30), list_290465, float_290466)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 22), list_290462, list_290465)
        # Adding element type (line 90)
        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_290467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        float_290468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 37), list_290467, float_290468)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 22), list_290462, list_290467)
        
        # Processing the call keyword arguments (line 90)
        kwargs_290469 = {}
        # Getting the type of 'np' (line 90)
        np_290460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 90)
        array_290461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 13), np_290460, 'array')
        # Calling array(args, kwargs) (line 90)
        array_call_result_290470 = invoke(stypy.reporting.localization.Localization(__file__, 90, 13), array_290461, *[list_290462], **kwargs_290469)
        
        # Assigning a type to the variable 'dc' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'dc', array_call_result_290470)
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        float_290471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'dt_requested', float_290471)
        
        # Assigning a BinOp to a Name (line 94):
        
        # Assigning a BinOp to a Name (line 94):
        float_290472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 19), 'float')
        
        # Call to eye(...): (line 94)
        # Processing the call arguments (line 94)
        int_290475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 32), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_290476 = {}
        # Getting the type of 'np' (line 94)
        np_290473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'np', False)
        # Obtaining the member 'eye' of a type (line 94)
        eye_290474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), np_290473, 'eye')
        # Calling eye(args, kwargs) (line 94)
        eye_call_result_290477 = invoke(stypy.reporting.localization.Localization(__file__, 94, 25), eye_290474, *[int_290475], **kwargs_290476)
        
        # Applying the binary operator '*' (line 94)
        result_mul_290478 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 19), '*', float_290472, eye_call_result_290477)
        
        # Assigning a type to the variable 'ad_truth' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'ad_truth', result_mul_290478)
        
        # Assigning a BinOp to a Name (line 95):
        
        # Assigning a BinOp to a Name (line 95):
        float_290479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'float')
        
        # Call to ones(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'tuple' (line 95)
        tuple_290482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 95)
        # Adding element type (line 95)
        int_290483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 34), tuple_290482, int_290483)
        # Adding element type (line 95)
        int_290484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 34), tuple_290482, int_290484)
        
        # Processing the call keyword arguments (line 95)
        kwargs_290485 = {}
        # Getting the type of 'np' (line 95)
        np_290480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'np', False)
        # Obtaining the member 'ones' of a type (line 95)
        ones_290481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), np_290480, 'ones')
        # Calling ones(args, kwargs) (line 95)
        ones_call_result_290486 = invoke(stypy.reporting.localization.Localization(__file__, 95, 25), ones_290481, *[tuple_290482], **kwargs_290485)
        
        # Applying the binary operator '*' (line 95)
        result_mul_290487 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 19), '*', float_290479, ones_call_result_290486)
        
        # Assigning a type to the variable 'bd_truth' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'bd_truth', result_mul_290487)
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to array(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_290490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_290491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        float_290492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 29), list_290491, float_290492)
        # Adding element type (line 96)
        float_290493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 29), list_290491, float_290493)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), list_290490, list_290491)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_290494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_290495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 29), list_290494, float_290495)
        # Adding element type (line 97)
        float_290496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 29), list_290494, float_290496)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), list_290490, list_290494)
        # Adding element type (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_290497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        # Adding element type (line 98)
        float_290498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 29), list_290497, float_290498)
        # Adding element type (line 98)
        float_290499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 29), list_290497, float_290499)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 28), list_290490, list_290497)
        
        # Processing the call keyword arguments (line 96)
        kwargs_290500 = {}
        # Getting the type of 'np' (line 96)
        np_290488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 96)
        array_290489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 19), np_290488, 'array')
        # Calling array(args, kwargs) (line 96)
        array_call_result_290501 = invoke(stypy.reporting.localization.Localization(__file__, 96, 19), array_290489, *[list_290490], **kwargs_290500)
        
        # Assigning a type to the variable 'cd_truth' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cd_truth', array_call_result_290501)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to array(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_290504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_290505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        float_290506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 29), list_290505, float_290506)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_290504, list_290505)
        # Adding element type (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_290507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        float_290508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 29), list_290507, float_290508)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_290504, list_290507)
        # Adding element type (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 101)
        list_290509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 101)
        # Adding element type (line 101)
        float_290510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 29), list_290509, float_290510)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), list_290504, list_290509)
        
        # Processing the call keyword arguments (line 99)
        kwargs_290511 = {}
        # Getting the type of 'np' (line 99)
        np_290502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 99)
        array_290503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 19), np_290502, 'array')
        # Calling array(args, kwargs) (line 99)
        array_call_result_290512 = invoke(stypy.reporting.localization.Localization(__file__, 99, 19), array_290503, *[list_290504], **kwargs_290511)
        
        # Assigning a type to the variable 'dd_truth' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'dd_truth', array_call_result_290512)
        
        # Assigning a Call to a Tuple (line 103):
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        int_290513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
        
        # Call to c2d(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_290515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'ac' (line 103)
        ac_290516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290515, ac_290516)
        # Adding element type (line 103)
        # Getting the type of 'bc' (line 103)
        bc_290517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290515, bc_290517)
        # Adding element type (line 103)
        # Getting the type of 'cc' (line 103)
        cc_290518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290515, cc_290518)
        # Adding element type (line 103)
        # Getting the type of 'dc' (line 103)
        dc_290519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290515, dc_290519)
        
        # Getting the type of 'dt_requested' (line 103)
        dt_requested_290520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 103)
        str_290521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'str', 'backward_diff')
        keyword_290522 = str_290521
        kwargs_290523 = {'method': keyword_290522}
        # Getting the type of 'c2d' (line 103)
        c2d_290514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 103)
        c2d_call_result_290524 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), c2d_290514, *[tuple_290515, dt_requested_290520], **kwargs_290523)
        
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___290525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), c2d_call_result_290524, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_290526 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___290525, int_290513)
        
        # Assigning a type to the variable 'tuple_var_assignment_289848' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289848', subscript_call_result_290526)
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        int_290527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
        
        # Call to c2d(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_290529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'ac' (line 103)
        ac_290530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290529, ac_290530)
        # Adding element type (line 103)
        # Getting the type of 'bc' (line 103)
        bc_290531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290529, bc_290531)
        # Adding element type (line 103)
        # Getting the type of 'cc' (line 103)
        cc_290532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290529, cc_290532)
        # Adding element type (line 103)
        # Getting the type of 'dc' (line 103)
        dc_290533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290529, dc_290533)
        
        # Getting the type of 'dt_requested' (line 103)
        dt_requested_290534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 103)
        str_290535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'str', 'backward_diff')
        keyword_290536 = str_290535
        kwargs_290537 = {'method': keyword_290536}
        # Getting the type of 'c2d' (line 103)
        c2d_290528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 103)
        c2d_call_result_290538 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), c2d_290528, *[tuple_290529, dt_requested_290534], **kwargs_290537)
        
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___290539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), c2d_call_result_290538, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_290540 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___290539, int_290527)
        
        # Assigning a type to the variable 'tuple_var_assignment_289849' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289849', subscript_call_result_290540)
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        int_290541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
        
        # Call to c2d(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_290543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'ac' (line 103)
        ac_290544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290543, ac_290544)
        # Adding element type (line 103)
        # Getting the type of 'bc' (line 103)
        bc_290545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290543, bc_290545)
        # Adding element type (line 103)
        # Getting the type of 'cc' (line 103)
        cc_290546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290543, cc_290546)
        # Adding element type (line 103)
        # Getting the type of 'dc' (line 103)
        dc_290547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290543, dc_290547)
        
        # Getting the type of 'dt_requested' (line 103)
        dt_requested_290548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 103)
        str_290549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'str', 'backward_diff')
        keyword_290550 = str_290549
        kwargs_290551 = {'method': keyword_290550}
        # Getting the type of 'c2d' (line 103)
        c2d_290542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 103)
        c2d_call_result_290552 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), c2d_290542, *[tuple_290543, dt_requested_290548], **kwargs_290551)
        
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___290553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), c2d_call_result_290552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_290554 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___290553, int_290541)
        
        # Assigning a type to the variable 'tuple_var_assignment_289850' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289850', subscript_call_result_290554)
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        int_290555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
        
        # Call to c2d(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_290557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'ac' (line 103)
        ac_290558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290557, ac_290558)
        # Adding element type (line 103)
        # Getting the type of 'bc' (line 103)
        bc_290559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290557, bc_290559)
        # Adding element type (line 103)
        # Getting the type of 'cc' (line 103)
        cc_290560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290557, cc_290560)
        # Adding element type (line 103)
        # Getting the type of 'dc' (line 103)
        dc_290561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290557, dc_290561)
        
        # Getting the type of 'dt_requested' (line 103)
        dt_requested_290562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 103)
        str_290563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'str', 'backward_diff')
        keyword_290564 = str_290563
        kwargs_290565 = {'method': keyword_290564}
        # Getting the type of 'c2d' (line 103)
        c2d_290556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 103)
        c2d_call_result_290566 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), c2d_290556, *[tuple_290557, dt_requested_290562], **kwargs_290565)
        
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___290567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), c2d_call_result_290566, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_290568 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___290567, int_290555)
        
        # Assigning a type to the variable 'tuple_var_assignment_289851' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289851', subscript_call_result_290568)
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        int_290569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'int')
        
        # Call to c2d(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_290571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        # Getting the type of 'ac' (line 103)
        ac_290572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290571, ac_290572)
        # Adding element type (line 103)
        # Getting the type of 'bc' (line 103)
        bc_290573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290571, bc_290573)
        # Adding element type (line 103)
        # Getting the type of 'cc' (line 103)
        cc_290574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290571, cc_290574)
        # Adding element type (line 103)
        # Getting the type of 'dc' (line 103)
        dc_290575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 34), tuple_290571, dc_290575)
        
        # Getting the type of 'dt_requested' (line 103)
        dt_requested_290576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 103)
        str_290577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 40), 'str', 'backward_diff')
        keyword_290578 = str_290577
        kwargs_290579 = {'method': keyword_290578}
        # Getting the type of 'c2d' (line 103)
        c2d_290570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 103)
        c2d_call_result_290580 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), c2d_290570, *[tuple_290571, dt_requested_290576], **kwargs_290579)
        
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___290581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), c2d_call_result_290580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_290582 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), getitem___290581, int_290569)
        
        # Assigning a type to the variable 'tuple_var_assignment_289852' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289852', subscript_call_result_290582)
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'tuple_var_assignment_289848' (line 103)
        tuple_var_assignment_289848_290583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289848')
        # Assigning a type to the variable 'ad' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'ad', tuple_var_assignment_289848_290583)
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'tuple_var_assignment_289849' (line 103)
        tuple_var_assignment_289849_290584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289849')
        # Assigning a type to the variable 'bd' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'bd', tuple_var_assignment_289849_290584)
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'tuple_var_assignment_289850' (line 103)
        tuple_var_assignment_289850_290585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289850')
        # Assigning a type to the variable 'cd' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'cd', tuple_var_assignment_289850_290585)
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'tuple_var_assignment_289851' (line 103)
        tuple_var_assignment_289851_290586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289851')
        # Assigning a type to the variable 'dd' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'dd', tuple_var_assignment_289851_290586)
        
        # Assigning a Name to a Name (line 103):
        # Getting the type of 'tuple_var_assignment_289852' (line 103)
        tuple_var_assignment_289852_290587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'tuple_var_assignment_289852')
        # Assigning a type to the variable 'dt' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'dt', tuple_var_assignment_289852_290587)
        
        # Call to assert_array_almost_equal(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'ad_truth' (line 106)
        ad_truth_290589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 106)
        ad_290590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 44), 'ad', False)
        # Processing the call keyword arguments (line 106)
        kwargs_290591 = {}
        # Getting the type of 'assert_array_almost_equal' (line 106)
        assert_array_almost_equal_290588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 106)
        assert_array_almost_equal_call_result_290592 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert_array_almost_equal_290588, *[ad_truth_290589, ad_290590], **kwargs_290591)
        
        
        # Call to assert_array_almost_equal(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'bd_truth' (line 107)
        bd_truth_290594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 107)
        bd_290595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'bd', False)
        # Processing the call keyword arguments (line 107)
        kwargs_290596 = {}
        # Getting the type of 'assert_array_almost_equal' (line 107)
        assert_array_almost_equal_290593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 107)
        assert_array_almost_equal_call_result_290597 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), assert_array_almost_equal_290593, *[bd_truth_290594, bd_290595], **kwargs_290596)
        
        
        # Call to assert_array_almost_equal(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'cd_truth' (line 108)
        cd_truth_290599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'cd_truth', False)
        # Getting the type of 'cd' (line 108)
        cd_290600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 44), 'cd', False)
        # Processing the call keyword arguments (line 108)
        kwargs_290601 = {}
        # Getting the type of 'assert_array_almost_equal' (line 108)
        assert_array_almost_equal_290598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 108)
        assert_array_almost_equal_call_result_290602 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), assert_array_almost_equal_290598, *[cd_truth_290599, cd_290600], **kwargs_290601)
        
        
        # Call to assert_array_almost_equal(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'dd_truth' (line 109)
        dd_truth_290604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 34), 'dd_truth', False)
        # Getting the type of 'dd' (line 109)
        dd_290605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'dd', False)
        # Processing the call keyword arguments (line 109)
        kwargs_290606 = {}
        # Getting the type of 'assert_array_almost_equal' (line 109)
        assert_array_almost_equal_290603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 109)
        assert_array_almost_equal_call_result_290607 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert_array_almost_equal_290603, *[dd_truth_290604, dd_290605], **kwargs_290606)
        
        
        # ################# End of 'test_backward_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_backward_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_290608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_backward_diff'
        return stypy_return_type_290608


    @norecursion
    def test_bilinear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bilinear'
        module_type_store = module_type_store.open_function_context('test_bilinear', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_bilinear')
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_bilinear.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_bilinear', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bilinear', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bilinear(...)' code ##################

        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to eye(...): (line 112)
        # Processing the call arguments (line 112)
        int_290611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 20), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_290612 = {}
        # Getting the type of 'np' (line 112)
        np_290609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'np', False)
        # Obtaining the member 'eye' of a type (line 112)
        eye_290610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), np_290609, 'eye')
        # Calling eye(args, kwargs) (line 112)
        eye_call_result_290613 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), eye_290610, *[int_290611], **kwargs_290612)
        
        # Assigning a type to the variable 'ac' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'ac', eye_call_result_290613)
        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        float_290614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'float')
        
        # Call to ones(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_290617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        int_290618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), tuple_290617, int_290618)
        # Adding element type (line 113)
        int_290619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 28), tuple_290617, int_290619)
        
        # Processing the call keyword arguments (line 113)
        kwargs_290620 = {}
        # Getting the type of 'np' (line 113)
        np_290615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'np', False)
        # Obtaining the member 'ones' of a type (line 113)
        ones_290616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), np_290615, 'ones')
        # Calling ones(args, kwargs) (line 113)
        ones_call_result_290621 = invoke(stypy.reporting.localization.Localization(__file__, 113, 19), ones_290616, *[tuple_290617], **kwargs_290620)
        
        # Applying the binary operator '*' (line 113)
        result_mul_290622 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 13), '*', float_290614, ones_call_result_290621)
        
        # Assigning a type to the variable 'bc' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'bc', result_mul_290622)
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to array(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_290625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_290626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_290627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 23), list_290626, float_290627)
        # Adding element type (line 114)
        float_290628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 23), list_290626, float_290628)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 22), list_290625, list_290626)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_290629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_290630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 36), list_290629, float_290630)
        # Adding element type (line 114)
        float_290631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 36), list_290629, float_290631)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 22), list_290625, list_290629)
        # Adding element type (line 114)
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_290632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        # Adding element type (line 114)
        float_290633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 48), list_290632, float_290633)
        # Adding element type (line 114)
        float_290634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 48), list_290632, float_290634)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 22), list_290625, list_290632)
        
        # Processing the call keyword arguments (line 114)
        kwargs_290635 = {}
        # Getting the type of 'np' (line 114)
        np_290623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 114)
        array_290624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 13), np_290623, 'array')
        # Calling array(args, kwargs) (line 114)
        array_call_result_290636 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), array_290624, *[list_290625], **kwargs_290635)
        
        # Assigning a type to the variable 'cc' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'cc', array_call_result_290636)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to array(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_290639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_290640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_290641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 23), list_290640, float_290641)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 22), list_290639, list_290640)
        # Adding element type (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_290642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_290643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 30), list_290642, float_290643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 22), list_290639, list_290642)
        # Adding element type (line 115)
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_290644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        # Adding element type (line 115)
        float_290645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 37), list_290644, float_290645)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 22), list_290639, list_290644)
        
        # Processing the call keyword arguments (line 115)
        kwargs_290646 = {}
        # Getting the type of 'np' (line 115)
        np_290637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 115)
        array_290638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 13), np_290637, 'array')
        # Calling array(args, kwargs) (line 115)
        array_call_result_290647 = invoke(stypy.reporting.localization.Localization(__file__, 115, 13), array_290638, *[list_290639], **kwargs_290646)
        
        # Assigning a type to the variable 'dc' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'dc', array_call_result_290647)
        
        # Assigning a Num to a Name (line 117):
        
        # Assigning a Num to a Name (line 117):
        float_290648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'dt_requested', float_290648)
        
        # Assigning a BinOp to a Name (line 119):
        
        # Assigning a BinOp to a Name (line 119):
        float_290649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 20), 'float')
        float_290650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'float')
        # Applying the binary operator 'div' (line 119)
        result_div_290651 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 20), 'div', float_290649, float_290650)
        
        
        # Call to eye(...): (line 119)
        # Processing the call arguments (line 119)
        int_290654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'int')
        # Processing the call keyword arguments (line 119)
        kwargs_290655 = {}
        # Getting the type of 'np' (line 119)
        np_290652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'np', False)
        # Obtaining the member 'eye' of a type (line 119)
        eye_290653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 33), np_290652, 'eye')
        # Calling eye(args, kwargs) (line 119)
        eye_call_result_290656 = invoke(stypy.reporting.localization.Localization(__file__, 119, 33), eye_290653, *[int_290654], **kwargs_290655)
        
        # Applying the binary operator '*' (line 119)
        result_mul_290657 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 19), '*', result_div_290651, eye_call_result_290656)
        
        # Assigning a type to the variable 'ad_truth' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'ad_truth', result_mul_290657)
        
        # Assigning a BinOp to a Name (line 120):
        
        # Assigning a BinOp to a Name (line 120):
        float_290658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 20), 'float')
        float_290659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 26), 'float')
        # Applying the binary operator 'div' (line 120)
        result_div_290660 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 20), 'div', float_290658, float_290659)
        
        
        # Call to ones(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_290663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        int_290664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_290663, int_290664)
        # Adding element type (line 120)
        int_290665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_290663, int_290665)
        
        # Processing the call keyword arguments (line 120)
        kwargs_290666 = {}
        # Getting the type of 'np' (line 120)
        np_290661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'np', False)
        # Obtaining the member 'ones' of a type (line 120)
        ones_290662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 33), np_290661, 'ones')
        # Calling ones(args, kwargs) (line 120)
        ones_call_result_290667 = invoke(stypy.reporting.localization.Localization(__file__, 120, 33), ones_290662, *[tuple_290663], **kwargs_290666)
        
        # Applying the binary operator '*' (line 120)
        result_mul_290668 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 19), '*', result_div_290660, ones_call_result_290667)
        
        # Assigning a type to the variable 'bd_truth' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'bd_truth', result_mul_290668)
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to array(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_290671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_290672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        float_290673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), list_290672, float_290673)
        # Adding element type (line 121)
        float_290674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'float')
        float_290675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 41), 'float')
        # Applying the binary operator 'div' (line 121)
        result_div_290676 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 35), 'div', float_290674, float_290675)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 29), list_290672, result_div_290676)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 28), list_290671, list_290672)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_290677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_290678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 30), 'float')
        float_290679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 36), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_290680 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 30), 'div', float_290678, float_290679)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 29), list_290677, result_div_290680)
        # Adding element type (line 122)
        float_290681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'float')
        float_290682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 47), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_290683 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 41), 'div', float_290681, float_290682)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 29), list_290677, result_div_290683)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 28), list_290671, list_290677)
        # Adding element type (line 121)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_290684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        float_290685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 30), 'float')
        float_290686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'float')
        # Applying the binary operator 'div' (line 123)
        result_div_290687 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 30), 'div', float_290685, float_290686)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), list_290684, result_div_290687)
        # Adding element type (line 123)
        float_290688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'float')
        float_290689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 47), 'float')
        # Applying the binary operator 'div' (line 123)
        result_div_290690 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 41), 'div', float_290688, float_290689)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 29), list_290684, result_div_290690)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 28), list_290671, list_290684)
        
        # Processing the call keyword arguments (line 121)
        kwargs_290691 = {}
        # Getting the type of 'np' (line 121)
        np_290669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 121)
        array_290670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), np_290669, 'array')
        # Calling array(args, kwargs) (line 121)
        array_call_result_290692 = invoke(stypy.reporting.localization.Localization(__file__, 121, 19), array_290670, *[list_290671], **kwargs_290691)
        
        # Assigning a type to the variable 'cd_truth' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'cd_truth', array_call_result_290692)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to array(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_290695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_290696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        # Adding element type (line 124)
        float_290697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 29), list_290696, float_290697)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 28), list_290695, list_290696)
        # Adding element type (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_290698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        # Adding element type (line 125)
        float_290699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'float')
        float_290700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 36), 'float')
        # Applying the binary operator 'div' (line 125)
        result_div_290701 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 30), 'div', float_290699, float_290700)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 29), list_290698, result_div_290701)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 28), list_290695, list_290698)
        # Adding element type (line 124)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_290702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        float_290703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 29), list_290702, float_290703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 28), list_290695, list_290702)
        
        # Processing the call keyword arguments (line 124)
        kwargs_290704 = {}
        # Getting the type of 'np' (line 124)
        np_290693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 124)
        array_290694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 19), np_290693, 'array')
        # Calling array(args, kwargs) (line 124)
        array_call_result_290705 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), array_290694, *[list_290695], **kwargs_290704)
        
        # Assigning a type to the variable 'dd_truth' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'dd_truth', array_call_result_290705)
        
        # Assigning a Call to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_290706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to c2d(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_290708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'ac' (line 128)
        ac_290709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290708, ac_290709)
        # Adding element type (line 128)
        # Getting the type of 'bc' (line 128)
        bc_290710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290708, bc_290710)
        # Adding element type (line 128)
        # Getting the type of 'cc' (line 128)
        cc_290711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290708, cc_290711)
        # Adding element type (line 128)
        # Getting the type of 'dc' (line 128)
        dc_290712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290708, dc_290712)
        
        # Getting the type of 'dt_requested' (line 128)
        dt_requested_290713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 128)
        str_290714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'bilinear')
        keyword_290715 = str_290714
        kwargs_290716 = {'method': keyword_290715}
        # Getting the type of 'c2d' (line 128)
        c2d_290707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 128)
        c2d_call_result_290717 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), c2d_290707, *[tuple_290708, dt_requested_290713], **kwargs_290716)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___290718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c2d_call_result_290717, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_290719 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___290718, int_290706)
        
        # Assigning a type to the variable 'tuple_var_assignment_289853' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289853', subscript_call_result_290719)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_290720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to c2d(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_290722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'ac' (line 128)
        ac_290723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290722, ac_290723)
        # Adding element type (line 128)
        # Getting the type of 'bc' (line 128)
        bc_290724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290722, bc_290724)
        # Adding element type (line 128)
        # Getting the type of 'cc' (line 128)
        cc_290725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290722, cc_290725)
        # Adding element type (line 128)
        # Getting the type of 'dc' (line 128)
        dc_290726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290722, dc_290726)
        
        # Getting the type of 'dt_requested' (line 128)
        dt_requested_290727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 128)
        str_290728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'bilinear')
        keyword_290729 = str_290728
        kwargs_290730 = {'method': keyword_290729}
        # Getting the type of 'c2d' (line 128)
        c2d_290721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 128)
        c2d_call_result_290731 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), c2d_290721, *[tuple_290722, dt_requested_290727], **kwargs_290730)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___290732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c2d_call_result_290731, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_290733 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___290732, int_290720)
        
        # Assigning a type to the variable 'tuple_var_assignment_289854' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289854', subscript_call_result_290733)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_290734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to c2d(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_290736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'ac' (line 128)
        ac_290737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290736, ac_290737)
        # Adding element type (line 128)
        # Getting the type of 'bc' (line 128)
        bc_290738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290736, bc_290738)
        # Adding element type (line 128)
        # Getting the type of 'cc' (line 128)
        cc_290739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290736, cc_290739)
        # Adding element type (line 128)
        # Getting the type of 'dc' (line 128)
        dc_290740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290736, dc_290740)
        
        # Getting the type of 'dt_requested' (line 128)
        dt_requested_290741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 128)
        str_290742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'bilinear')
        keyword_290743 = str_290742
        kwargs_290744 = {'method': keyword_290743}
        # Getting the type of 'c2d' (line 128)
        c2d_290735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 128)
        c2d_call_result_290745 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), c2d_290735, *[tuple_290736, dt_requested_290741], **kwargs_290744)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___290746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c2d_call_result_290745, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_290747 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___290746, int_290734)
        
        # Assigning a type to the variable 'tuple_var_assignment_289855' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289855', subscript_call_result_290747)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_290748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to c2d(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_290750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'ac' (line 128)
        ac_290751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290750, ac_290751)
        # Adding element type (line 128)
        # Getting the type of 'bc' (line 128)
        bc_290752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290750, bc_290752)
        # Adding element type (line 128)
        # Getting the type of 'cc' (line 128)
        cc_290753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290750, cc_290753)
        # Adding element type (line 128)
        # Getting the type of 'dc' (line 128)
        dc_290754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290750, dc_290754)
        
        # Getting the type of 'dt_requested' (line 128)
        dt_requested_290755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 128)
        str_290756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'bilinear')
        keyword_290757 = str_290756
        kwargs_290758 = {'method': keyword_290757}
        # Getting the type of 'c2d' (line 128)
        c2d_290749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 128)
        c2d_call_result_290759 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), c2d_290749, *[tuple_290750, dt_requested_290755], **kwargs_290758)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___290760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c2d_call_result_290759, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_290761 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___290760, int_290748)
        
        # Assigning a type to the variable 'tuple_var_assignment_289856' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289856', subscript_call_result_290761)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_290762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to c2d(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'tuple' (line 128)
        tuple_290764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 128)
        # Adding element type (line 128)
        # Getting the type of 'ac' (line 128)
        ac_290765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290764, ac_290765)
        # Adding element type (line 128)
        # Getting the type of 'bc' (line 128)
        bc_290766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290764, bc_290766)
        # Adding element type (line 128)
        # Getting the type of 'cc' (line 128)
        cc_290767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290764, cc_290767)
        # Adding element type (line 128)
        # Getting the type of 'dc' (line 128)
        dc_290768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 34), tuple_290764, dc_290768)
        
        # Getting the type of 'dt_requested' (line 128)
        dt_requested_290769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 128)
        str_290770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'bilinear')
        keyword_290771 = str_290770
        kwargs_290772 = {'method': keyword_290771}
        # Getting the type of 'c2d' (line 128)
        c2d_290763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 128)
        c2d_call_result_290773 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), c2d_290763, *[tuple_290764, dt_requested_290769], **kwargs_290772)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___290774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), c2d_call_result_290773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_290775 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___290774, int_290762)
        
        # Assigning a type to the variable 'tuple_var_assignment_289857' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289857', subscript_call_result_290775)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_289853' (line 128)
        tuple_var_assignment_289853_290776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289853')
        # Assigning a type to the variable 'ad' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'ad', tuple_var_assignment_289853_290776)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_289854' (line 128)
        tuple_var_assignment_289854_290777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289854')
        # Assigning a type to the variable 'bd' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'bd', tuple_var_assignment_289854_290777)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_289855' (line 128)
        tuple_var_assignment_289855_290778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289855')
        # Assigning a type to the variable 'cd' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'cd', tuple_var_assignment_289855_290778)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_289856' (line 128)
        tuple_var_assignment_289856_290779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289856')
        # Assigning a type to the variable 'dd' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'dd', tuple_var_assignment_289856_290779)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_289857' (line 128)
        tuple_var_assignment_289857_290780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_289857')
        # Assigning a type to the variable 'dt' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'dt', tuple_var_assignment_289857_290780)
        
        # Call to assert_array_almost_equal(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'ad_truth' (line 131)
        ad_truth_290782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 131)
        ad_290783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 44), 'ad', False)
        # Processing the call keyword arguments (line 131)
        kwargs_290784 = {}
        # Getting the type of 'assert_array_almost_equal' (line 131)
        assert_array_almost_equal_290781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 131)
        assert_array_almost_equal_call_result_290785 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assert_array_almost_equal_290781, *[ad_truth_290782, ad_290783], **kwargs_290784)
        
        
        # Call to assert_array_almost_equal(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'bd_truth' (line 132)
        bd_truth_290787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 132)
        bd_290788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'bd', False)
        # Processing the call keyword arguments (line 132)
        kwargs_290789 = {}
        # Getting the type of 'assert_array_almost_equal' (line 132)
        assert_array_almost_equal_290786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 132)
        assert_array_almost_equal_call_result_290790 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assert_array_almost_equal_290786, *[bd_truth_290787, bd_290788], **kwargs_290789)
        
        
        # Call to assert_array_almost_equal(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'cd_truth' (line 133)
        cd_truth_290792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'cd_truth', False)
        # Getting the type of 'cd' (line 133)
        cd_290793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 44), 'cd', False)
        # Processing the call keyword arguments (line 133)
        kwargs_290794 = {}
        # Getting the type of 'assert_array_almost_equal' (line 133)
        assert_array_almost_equal_290791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 133)
        assert_array_almost_equal_call_result_290795 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), assert_array_almost_equal_290791, *[cd_truth_290792, cd_290793], **kwargs_290794)
        
        
        # Call to assert_array_almost_equal(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'dd_truth' (line 134)
        dd_truth_290797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 34), 'dd_truth', False)
        # Getting the type of 'dd' (line 134)
        dd_290798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'dd', False)
        # Processing the call keyword arguments (line 134)
        kwargs_290799 = {}
        # Getting the type of 'assert_array_almost_equal' (line 134)
        assert_array_almost_equal_290796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 134)
        assert_array_almost_equal_call_result_290800 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), assert_array_almost_equal_290796, *[dd_truth_290797, dd_290798], **kwargs_290799)
        
        
        # Call to assert_almost_equal(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'dt_requested' (line 135)
        dt_requested_290802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 135)
        dt_290803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 42), 'dt', False)
        # Processing the call keyword arguments (line 135)
        kwargs_290804 = {}
        # Getting the type of 'assert_almost_equal' (line 135)
        assert_almost_equal_290801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 135)
        assert_almost_equal_call_result_290805 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assert_almost_equal_290801, *[dt_requested_290802, dt_290803], **kwargs_290804)
        
        
        # Assigning a BinOp to a Name (line 139):
        
        # Assigning a BinOp to a Name (line 139):
        float_290806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 19), 'float')
        
        # Call to eye(...): (line 139)
        # Processing the call arguments (line 139)
        int_290809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 32), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_290810 = {}
        # Getting the type of 'np' (line 139)
        np_290807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'np', False)
        # Obtaining the member 'eye' of a type (line 139)
        eye_290808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), np_290807, 'eye')
        # Calling eye(args, kwargs) (line 139)
        eye_call_result_290811 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), eye_290808, *[int_290809], **kwargs_290810)
        
        # Applying the binary operator '*' (line 139)
        result_mul_290812 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 19), '*', float_290806, eye_call_result_290811)
        
        # Assigning a type to the variable 'ad_truth' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'ad_truth', result_mul_290812)
        
        # Assigning a BinOp to a Name (line 140):
        
        # Assigning a BinOp to a Name (line 140):
        float_290813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 19), 'float')
        
        # Call to ones(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_290816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        int_290817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 34), tuple_290816, int_290817)
        # Adding element type (line 140)
        int_290818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 34), tuple_290816, int_290818)
        
        # Processing the call keyword arguments (line 140)
        kwargs_290819 = {}
        # Getting the type of 'np' (line 140)
        np_290814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'np', False)
        # Obtaining the member 'ones' of a type (line 140)
        ones_290815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 25), np_290814, 'ones')
        # Calling ones(args, kwargs) (line 140)
        ones_call_result_290820 = invoke(stypy.reporting.localization.Localization(__file__, 140, 25), ones_290815, *[tuple_290816], **kwargs_290819)
        
        # Applying the binary operator '*' (line 140)
        result_mul_290821 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 19), '*', float_290813, ones_call_result_290820)
        
        # Assigning a type to the variable 'bd_truth' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'bd_truth', result_mul_290821)
        
        # Assigning a Call to a Name (line 141):
        
        # Assigning a Call to a Name (line 141):
        
        # Call to array(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_290824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_290825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        float_290826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 29), list_290825, float_290826)
        # Adding element type (line 141)
        float_290827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 29), list_290825, float_290827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 28), list_290824, list_290825)
        # Adding element type (line 141)
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_290828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        float_290829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 41), list_290828, float_290829)
        # Adding element type (line 141)
        float_290830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 41), list_290828, float_290830)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 28), list_290824, list_290828)
        # Adding element type (line 141)
        
        # Obtaining an instance of the builtin type 'list' (line 141)
        list_290831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 141)
        # Adding element type (line 141)
        float_290832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 53), list_290831, float_290832)
        # Adding element type (line 141)
        float_290833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 59), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 53), list_290831, float_290833)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 28), list_290824, list_290831)
        
        # Processing the call keyword arguments (line 141)
        kwargs_290834 = {}
        # Getting the type of 'np' (line 141)
        np_290822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 141)
        array_290823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), np_290822, 'array')
        # Calling array(args, kwargs) (line 141)
        array_call_result_290835 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), array_290823, *[list_290824], **kwargs_290834)
        
        # Assigning a type to the variable 'cd_truth' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'cd_truth', array_call_result_290835)
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to array(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_290838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_290839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        float_290840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 29), list_290839, float_290840)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 28), list_290838, list_290839)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_290841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        float_290842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 38), list_290841, float_290842)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 28), list_290838, list_290841)
        # Adding element type (line 142)
        
        # Obtaining an instance of the builtin type 'list' (line 142)
        list_290843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 142)
        # Adding element type (line 142)
        float_290844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 45), list_290843, float_290844)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 28), list_290838, list_290843)
        
        # Processing the call keyword arguments (line 142)
        kwargs_290845 = {}
        # Getting the type of 'np' (line 142)
        np_290836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 142)
        array_290837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), np_290836, 'array')
        # Calling array(args, kwargs) (line 142)
        array_call_result_290846 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), array_290837, *[list_290838], **kwargs_290845)
        
        # Assigning a type to the variable 'dd_truth' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'dd_truth', array_call_result_290846)
        
        # Assigning a BinOp to a Name (line 144):
        
        # Assigning a BinOp to a Name (line 144):
        float_290847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'float')
        float_290848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 29), 'float')
        # Applying the binary operator 'div' (line 144)
        result_div_290849 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 23), 'div', float_290847, float_290848)
        
        # Assigning a type to the variable 'dt_requested' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'dt_requested', result_div_290849)
        
        # Assigning a Call to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_290850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to c2d(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_290852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'ac' (line 146)
        ac_290853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290852, ac_290853)
        # Adding element type (line 146)
        # Getting the type of 'bc' (line 146)
        bc_290854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290852, bc_290854)
        # Adding element type (line 146)
        # Getting the type of 'cc' (line 146)
        cc_290855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290852, cc_290855)
        # Adding element type (line 146)
        # Getting the type of 'dc' (line 146)
        dc_290856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290852, dc_290856)
        
        # Getting the type of 'dt_requested' (line 146)
        dt_requested_290857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 146)
        str_290858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'str', 'bilinear')
        keyword_290859 = str_290858
        kwargs_290860 = {'method': keyword_290859}
        # Getting the type of 'c2d' (line 146)
        c2d_290851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 146)
        c2d_call_result_290861 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), c2d_290851, *[tuple_290852, dt_requested_290857], **kwargs_290860)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___290862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c2d_call_result_290861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_290863 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___290862, int_290850)
        
        # Assigning a type to the variable 'tuple_var_assignment_289858' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289858', subscript_call_result_290863)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_290864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to c2d(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_290866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'ac' (line 146)
        ac_290867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290866, ac_290867)
        # Adding element type (line 146)
        # Getting the type of 'bc' (line 146)
        bc_290868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290866, bc_290868)
        # Adding element type (line 146)
        # Getting the type of 'cc' (line 146)
        cc_290869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290866, cc_290869)
        # Adding element type (line 146)
        # Getting the type of 'dc' (line 146)
        dc_290870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290866, dc_290870)
        
        # Getting the type of 'dt_requested' (line 146)
        dt_requested_290871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 146)
        str_290872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'str', 'bilinear')
        keyword_290873 = str_290872
        kwargs_290874 = {'method': keyword_290873}
        # Getting the type of 'c2d' (line 146)
        c2d_290865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 146)
        c2d_call_result_290875 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), c2d_290865, *[tuple_290866, dt_requested_290871], **kwargs_290874)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___290876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c2d_call_result_290875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_290877 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___290876, int_290864)
        
        # Assigning a type to the variable 'tuple_var_assignment_289859' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289859', subscript_call_result_290877)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_290878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to c2d(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_290880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'ac' (line 146)
        ac_290881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290880, ac_290881)
        # Adding element type (line 146)
        # Getting the type of 'bc' (line 146)
        bc_290882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290880, bc_290882)
        # Adding element type (line 146)
        # Getting the type of 'cc' (line 146)
        cc_290883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290880, cc_290883)
        # Adding element type (line 146)
        # Getting the type of 'dc' (line 146)
        dc_290884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290880, dc_290884)
        
        # Getting the type of 'dt_requested' (line 146)
        dt_requested_290885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 146)
        str_290886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'str', 'bilinear')
        keyword_290887 = str_290886
        kwargs_290888 = {'method': keyword_290887}
        # Getting the type of 'c2d' (line 146)
        c2d_290879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 146)
        c2d_call_result_290889 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), c2d_290879, *[tuple_290880, dt_requested_290885], **kwargs_290888)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___290890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c2d_call_result_290889, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_290891 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___290890, int_290878)
        
        # Assigning a type to the variable 'tuple_var_assignment_289860' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289860', subscript_call_result_290891)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_290892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to c2d(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_290894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'ac' (line 146)
        ac_290895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290894, ac_290895)
        # Adding element type (line 146)
        # Getting the type of 'bc' (line 146)
        bc_290896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290894, bc_290896)
        # Adding element type (line 146)
        # Getting the type of 'cc' (line 146)
        cc_290897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290894, cc_290897)
        # Adding element type (line 146)
        # Getting the type of 'dc' (line 146)
        dc_290898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290894, dc_290898)
        
        # Getting the type of 'dt_requested' (line 146)
        dt_requested_290899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 146)
        str_290900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'str', 'bilinear')
        keyword_290901 = str_290900
        kwargs_290902 = {'method': keyword_290901}
        # Getting the type of 'c2d' (line 146)
        c2d_290893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 146)
        c2d_call_result_290903 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), c2d_290893, *[tuple_290894, dt_requested_290899], **kwargs_290902)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___290904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c2d_call_result_290903, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_290905 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___290904, int_290892)
        
        # Assigning a type to the variable 'tuple_var_assignment_289861' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289861', subscript_call_result_290905)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_290906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'int')
        
        # Call to c2d(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_290908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'ac' (line 146)
        ac_290909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'ac', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290908, ac_290909)
        # Adding element type (line 146)
        # Getting the type of 'bc' (line 146)
        bc_290910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'bc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290908, bc_290910)
        # Adding element type (line 146)
        # Getting the type of 'cc' (line 146)
        cc_290911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 42), 'cc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290908, cc_290911)
        # Adding element type (line 146)
        # Getting the type of 'dc' (line 146)
        dc_290912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'dc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 34), tuple_290908, dc_290912)
        
        # Getting the type of 'dt_requested' (line 146)
        dt_requested_290913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'dt_requested', False)
        # Processing the call keyword arguments (line 146)
        str_290914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 40), 'str', 'bilinear')
        keyword_290915 = str_290914
        kwargs_290916 = {'method': keyword_290915}
        # Getting the type of 'c2d' (line 146)
        c2d_290907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 146)
        c2d_call_result_290917 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), c2d_290907, *[tuple_290908, dt_requested_290913], **kwargs_290916)
        
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___290918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), c2d_call_result_290917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_290919 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), getitem___290918, int_290906)
        
        # Assigning a type to the variable 'tuple_var_assignment_289862' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289862', subscript_call_result_290919)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_289858' (line 146)
        tuple_var_assignment_289858_290920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289858')
        # Assigning a type to the variable 'ad' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'ad', tuple_var_assignment_289858_290920)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_289859' (line 146)
        tuple_var_assignment_289859_290921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289859')
        # Assigning a type to the variable 'bd' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'bd', tuple_var_assignment_289859_290921)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_289860' (line 146)
        tuple_var_assignment_289860_290922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289860')
        # Assigning a type to the variable 'cd' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'cd', tuple_var_assignment_289860_290922)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_289861' (line 146)
        tuple_var_assignment_289861_290923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289861')
        # Assigning a type to the variable 'dd' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'dd', tuple_var_assignment_289861_290923)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_289862' (line 146)
        tuple_var_assignment_289862_290924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'tuple_var_assignment_289862')
        # Assigning a type to the variable 'dt' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'dt', tuple_var_assignment_289862_290924)
        
        # Call to assert_array_almost_equal(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'ad_truth' (line 149)
        ad_truth_290926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'ad_truth', False)
        # Getting the type of 'ad' (line 149)
        ad_290927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 44), 'ad', False)
        # Processing the call keyword arguments (line 149)
        kwargs_290928 = {}
        # Getting the type of 'assert_array_almost_equal' (line 149)
        assert_array_almost_equal_290925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 149)
        assert_array_almost_equal_call_result_290929 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_array_almost_equal_290925, *[ad_truth_290926, ad_290927], **kwargs_290928)
        
        
        # Call to assert_array_almost_equal(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'bd_truth' (line 150)
        bd_truth_290931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 'bd_truth', False)
        # Getting the type of 'bd' (line 150)
        bd_290932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'bd', False)
        # Processing the call keyword arguments (line 150)
        kwargs_290933 = {}
        # Getting the type of 'assert_array_almost_equal' (line 150)
        assert_array_almost_equal_290930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 150)
        assert_array_almost_equal_call_result_290934 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assert_array_almost_equal_290930, *[bd_truth_290931, bd_290932], **kwargs_290933)
        
        
        # Call to assert_array_almost_equal(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'cd_truth' (line 151)
        cd_truth_290936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'cd_truth', False)
        # Getting the type of 'cd' (line 151)
        cd_290937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 44), 'cd', False)
        # Processing the call keyword arguments (line 151)
        kwargs_290938 = {}
        # Getting the type of 'assert_array_almost_equal' (line 151)
        assert_array_almost_equal_290935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 151)
        assert_array_almost_equal_call_result_290939 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), assert_array_almost_equal_290935, *[cd_truth_290936, cd_290937], **kwargs_290938)
        
        
        # Call to assert_array_almost_equal(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'dd_truth' (line 152)
        dd_truth_290941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 34), 'dd_truth', False)
        # Getting the type of 'dd' (line 152)
        dd_290942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'dd', False)
        # Processing the call keyword arguments (line 152)
        kwargs_290943 = {}
        # Getting the type of 'assert_array_almost_equal' (line 152)
        assert_array_almost_equal_290940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 152)
        assert_array_almost_equal_call_result_290944 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), assert_array_almost_equal_290940, *[dd_truth_290941, dd_290942], **kwargs_290943)
        
        
        # Call to assert_almost_equal(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'dt_requested' (line 153)
        dt_requested_290946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 153)
        dt_290947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 42), 'dt', False)
        # Processing the call keyword arguments (line 153)
        kwargs_290948 = {}
        # Getting the type of 'assert_almost_equal' (line 153)
        assert_almost_equal_290945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 153)
        assert_almost_equal_call_result_290949 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_almost_equal_290945, *[dt_requested_290946, dt_290947], **kwargs_290948)
        
        
        # ################# End of 'test_bilinear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bilinear' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_290950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290950)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bilinear'
        return stypy_return_type_290950


    @norecursion
    def test_transferfunction(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_transferfunction'
        module_type_store = module_type_store.open_function_context('test_transferfunction', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_transferfunction')
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_transferfunction.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_transferfunction', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_transferfunction', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_transferfunction(...)' code ##################

        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to array(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_290953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        float_290954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_290953, float_290954)
        # Adding element type (line 156)
        float_290955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_290953, float_290955)
        # Adding element type (line 156)
        float_290956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), list_290953, float_290956)
        
        # Processing the call keyword arguments (line 156)
        kwargs_290957 = {}
        # Getting the type of 'np' (line 156)
        np_290951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 156)
        array_290952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), np_290951, 'array')
        # Calling array(args, kwargs) (line 156)
        array_call_result_290958 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), array_290952, *[list_290953], **kwargs_290957)
        
        # Assigning a type to the variable 'numc' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'numc', array_call_result_290958)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to array(...): (line 157)
        # Processing the call arguments (line 157)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_290961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        float_290962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 24), list_290961, float_290962)
        # Adding element type (line 157)
        float_290963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 24), list_290961, float_290963)
        # Adding element type (line 157)
        float_290964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 24), list_290961, float_290964)
        
        # Processing the call keyword arguments (line 157)
        kwargs_290965 = {}
        # Getting the type of 'np' (line 157)
        np_290959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 157)
        array_290960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 15), np_290959, 'array')
        # Calling array(args, kwargs) (line 157)
        array_call_result_290966 = invoke(stypy.reporting.localization.Localization(__file__, 157, 15), array_290960, *[list_290961], **kwargs_290965)
        
        # Assigning a type to the variable 'denc' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'denc', array_call_result_290966)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to array(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_290969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_290970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        float_290971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'float')
        float_290972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'float')
        # Applying the binary operator 'div' (line 159)
        result_div_290973 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 26), 'div', float_290971, float_290972)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 25), list_290970, result_div_290973)
        # Adding element type (line 159)
        float_290974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 25), list_290970, float_290974)
        # Adding element type (line 159)
        float_290975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 25), list_290970, float_290975)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 24), list_290969, list_290970)
        
        # Processing the call keyword arguments (line 159)
        kwargs_290976 = {}
        # Getting the type of 'np' (line 159)
        np_290967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 159)
        array_290968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), np_290967, 'array')
        # Calling array(args, kwargs) (line 159)
        array_call_result_290977 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), array_290968, *[list_290969], **kwargs_290976)
        
        # Assigning a type to the variable 'numd' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'numd', array_call_result_290977)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to array(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_290980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        float_290981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 24), list_290980, float_290981)
        # Adding element type (line 160)
        float_290982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 24), list_290980, float_290982)
        # Adding element type (line 160)
        float_290983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 24), list_290980, float_290983)
        
        # Processing the call keyword arguments (line 160)
        kwargs_290984 = {}
        # Getting the type of 'np' (line 160)
        np_290978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'np', False)
        # Obtaining the member 'array' of a type (line 160)
        array_290979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), np_290978, 'array')
        # Calling array(args, kwargs) (line 160)
        array_call_result_290985 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), array_290979, *[list_290980], **kwargs_290984)
        
        # Assigning a type to the variable 'dend' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'dend', array_call_result_290985)
        
        # Assigning a Num to a Name (line 162):
        
        # Assigning a Num to a Name (line 162):
        float_290986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'dt_requested', float_290986)
        
        # Assigning a Call to a Tuple (line 164):
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_290987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        
        # Call to c2d(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_290989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'numc' (line 164)
        numc_290990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'numc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_290989, numc_290990)
        # Adding element type (line 164)
        # Getting the type of 'denc' (line 164)
        denc_290991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'denc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_290989, denc_290991)
        
        # Getting the type of 'dt_requested' (line 164)
        dt_requested_290992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'dt_requested', False)
        # Processing the call keyword arguments (line 164)
        str_290993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 62), 'str', 'zoh')
        keyword_290994 = str_290993
        kwargs_290995 = {'method': keyword_290994}
        # Getting the type of 'c2d' (line 164)
        c2d_290988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 164)
        c2d_call_result_290996 = invoke(stypy.reporting.localization.Localization(__file__, 164, 23), c2d_290988, *[tuple_290989, dt_requested_290992], **kwargs_290995)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___290997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), c2d_call_result_290996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_290998 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___290997, int_290987)
        
        # Assigning a type to the variable 'tuple_var_assignment_289863' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289863', subscript_call_result_290998)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_290999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        
        # Call to c2d(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_291001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'numc' (line 164)
        numc_291002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'numc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_291001, numc_291002)
        # Adding element type (line 164)
        # Getting the type of 'denc' (line 164)
        denc_291003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'denc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_291001, denc_291003)
        
        # Getting the type of 'dt_requested' (line 164)
        dt_requested_291004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'dt_requested', False)
        # Processing the call keyword arguments (line 164)
        str_291005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 62), 'str', 'zoh')
        keyword_291006 = str_291005
        kwargs_291007 = {'method': keyword_291006}
        # Getting the type of 'c2d' (line 164)
        c2d_291000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 164)
        c2d_call_result_291008 = invoke(stypy.reporting.localization.Localization(__file__, 164, 23), c2d_291000, *[tuple_291001, dt_requested_291004], **kwargs_291007)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___291009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), c2d_call_result_291008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_291010 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___291009, int_290999)
        
        # Assigning a type to the variable 'tuple_var_assignment_289864' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289864', subscript_call_result_291010)
        
        # Assigning a Subscript to a Name (line 164):
        
        # Obtaining the type of the subscript
        int_291011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
        
        # Call to c2d(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Obtaining an instance of the builtin type 'tuple' (line 164)
        tuple_291013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 164)
        # Adding element type (line 164)
        # Getting the type of 'numc' (line 164)
        numc_291014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 28), 'numc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_291013, numc_291014)
        # Adding element type (line 164)
        # Getting the type of 'denc' (line 164)
        denc_291015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'denc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 28), tuple_291013, denc_291015)
        
        # Getting the type of 'dt_requested' (line 164)
        dt_requested_291016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 41), 'dt_requested', False)
        # Processing the call keyword arguments (line 164)
        str_291017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 62), 'str', 'zoh')
        keyword_291018 = str_291017
        kwargs_291019 = {'method': keyword_291018}
        # Getting the type of 'c2d' (line 164)
        c2d_291012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 164)
        c2d_call_result_291020 = invoke(stypy.reporting.localization.Localization(__file__, 164, 23), c2d_291012, *[tuple_291013, dt_requested_291016], **kwargs_291019)
        
        # Obtaining the member '__getitem__' of a type (line 164)
        getitem___291021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), c2d_call_result_291020, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 164)
        subscript_call_result_291022 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___291021, int_291011)
        
        # Assigning a type to the variable 'tuple_var_assignment_289865' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289865', subscript_call_result_291022)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_289863' (line 164)
        tuple_var_assignment_289863_291023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289863')
        # Assigning a type to the variable 'num' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'num', tuple_var_assignment_289863_291023)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_289864' (line 164)
        tuple_var_assignment_289864_291024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289864')
        # Assigning a type to the variable 'den' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'den', tuple_var_assignment_289864_291024)
        
        # Assigning a Name to a Name (line 164):
        # Getting the type of 'tuple_var_assignment_289865' (line 164)
        tuple_var_assignment_289865_291025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_289865')
        # Assigning a type to the variable 'dt' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'dt', tuple_var_assignment_289865_291025)
        
        # Call to assert_array_almost_equal(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'numd' (line 166)
        numd_291027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 34), 'numd', False)
        # Getting the type of 'num' (line 166)
        num_291028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'num', False)
        # Processing the call keyword arguments (line 166)
        kwargs_291029 = {}
        # Getting the type of 'assert_array_almost_equal' (line 166)
        assert_array_almost_equal_291026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 166)
        assert_array_almost_equal_call_result_291030 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), assert_array_almost_equal_291026, *[numd_291027, num_291028], **kwargs_291029)
        
        
        # Call to assert_array_almost_equal(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'dend' (line 167)
        dend_291032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 34), 'dend', False)
        # Getting the type of 'den' (line 167)
        den_291033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 40), 'den', False)
        # Processing the call keyword arguments (line 167)
        kwargs_291034 = {}
        # Getting the type of 'assert_array_almost_equal' (line 167)
        assert_array_almost_equal_291031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 167)
        assert_array_almost_equal_call_result_291035 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), assert_array_almost_equal_291031, *[dend_291032, den_291033], **kwargs_291034)
        
        
        # Call to assert_almost_equal(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'dt_requested' (line 168)
        dt_requested_291037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 168)
        dt_291038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'dt', False)
        # Processing the call keyword arguments (line 168)
        kwargs_291039 = {}
        # Getting the type of 'assert_almost_equal' (line 168)
        assert_almost_equal_291036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 168)
        assert_almost_equal_call_result_291040 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), assert_almost_equal_291036, *[dt_requested_291037, dt_291038], **kwargs_291039)
        
        
        # ################# End of 'test_transferfunction(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_transferfunction' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_291041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_transferfunction'
        return stypy_return_type_291041


    @norecursion
    def test_zerospolesgain(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zerospolesgain'
        module_type_store = module_type_store.open_function_context('test_zerospolesgain', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_zerospolesgain')
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_zerospolesgain.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_zerospolesgain', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zerospolesgain', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zerospolesgain(...)' code ##################

        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to array(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_291044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        float_291045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), list_291044, float_291045)
        # Adding element type (line 171)
        float_291046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 27), list_291044, float_291046)
        
        # Processing the call keyword arguments (line 171)
        kwargs_291047 = {}
        # Getting the type of 'np' (line 171)
        np_291042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 171)
        array_291043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 18), np_291042, 'array')
        # Calling array(args, kwargs) (line 171)
        array_call_result_291048 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), array_291043, *[list_291044], **kwargs_291047)
        
        # Assigning a type to the variable 'zeros_c' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'zeros_c', array_call_result_291048)
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to array(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_291051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        complex_291052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'complex')
        
        # Call to sqrt(...): (line 172)
        # Processing the call arguments (line 172)
        int_291055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 42), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_291056 = {}
        # Getting the type of 'np' (line 172)
        np_291053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 172)
        sqrt_291054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 34), np_291053, 'sqrt')
        # Calling sqrt(args, kwargs) (line 172)
        sqrt_call_result_291057 = invoke(stypy.reporting.localization.Localization(__file__, 172, 34), sqrt_291054, *[int_291055], **kwargs_291056)
        
        # Applying the binary operator 'div' (line 172)
        result_div_291058 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 28), 'div', complex_291052, sqrt_call_result_291057)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 27), list_291051, result_div_291058)
        # Adding element type (line 172)
        complex_291059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 46), 'complex')
        
        # Call to sqrt(...): (line 172)
        # Processing the call arguments (line 172)
        int_291062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 61), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_291063 = {}
        # Getting the type of 'np' (line 172)
        np_291060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 53), 'np', False)
        # Obtaining the member 'sqrt' of a type (line 172)
        sqrt_291061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 53), np_291060, 'sqrt')
        # Calling sqrt(args, kwargs) (line 172)
        sqrt_call_result_291064 = invoke(stypy.reporting.localization.Localization(__file__, 172, 53), sqrt_291061, *[int_291062], **kwargs_291063)
        
        # Applying the binary operator 'div' (line 172)
        result_div_291065 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 46), 'div', complex_291059, sqrt_call_result_291064)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 27), list_291051, result_div_291065)
        
        # Processing the call keyword arguments (line 172)
        kwargs_291066 = {}
        # Getting the type of 'np' (line 172)
        np_291049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 172)
        array_291050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 18), np_291049, 'array')
        # Calling array(args, kwargs) (line 172)
        array_call_result_291067 = invoke(stypy.reporting.localization.Localization(__file__, 172, 18), array_291050, *[list_291051], **kwargs_291066)
        
        # Assigning a type to the variable 'poles_c' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'poles_c', array_call_result_291067)
        
        # Assigning a Num to a Name (line 173):
        
        # Assigning a Num to a Name (line 173):
        float_291068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 14), 'float')
        # Assigning a type to the variable 'k_c' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'k_c', float_291068)
        
        # Assigning a List to a Name (line 175):
        
        # Assigning a List to a Name (line 175):
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_291069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        float_291070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), list_291069, float_291070)
        # Adding element type (line 175)
        float_291071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), list_291069, float_291071)
        
        # Assigning a type to the variable 'zeros_d' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'zeros_d', list_291069)
        
        # Assigning a List to a Name (line 176):
        
        # Assigning a List to a Name (line 176):
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_291072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        float_291073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'float')
        complex_291074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 39), 'complex')
        # Applying the binary operator '+' (line 176)
        result_add_291075 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 19), '+', float_291073, complex_291074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_291072, result_add_291075)
        # Adding element type (line 176)
        float_291076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 19), 'float')
        complex_291077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 39), 'complex')
        # Applying the binary operator '-' (line 177)
        result_sub_291078 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 19), '-', float_291076, complex_291077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 18), list_291072, result_sub_291078)
        
        # Assigning a type to the variable 'polls_d' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'polls_d', list_291072)
        
        # Assigning a Num to a Name (line 178):
        
        # Assigning a Num to a Name (line 178):
        float_291079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 14), 'float')
        # Assigning a type to the variable 'k_d' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'k_d', float_291079)
        
        # Assigning a Num to a Name (line 180):
        
        # Assigning a Num to a Name (line 180):
        float_291080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'float')
        # Assigning a type to the variable 'dt_requested' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'dt_requested', float_291080)
        
        # Assigning a Call to a Tuple (line 182):
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_291081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to c2d(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_291083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'zeros_c' (line 182)
        zeros_c_291084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'zeros_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291083, zeros_c_291084)
        # Adding element type (line 182)
        # Getting the type of 'poles_c' (line 182)
        poles_c_291085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'poles_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291083, poles_c_291085)
        # Adding element type (line 182)
        # Getting the type of 'k_c' (line 182)
        k_c_291086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'k_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291083, k_c_291086)
        
        # Getting the type of 'dt_requested' (line 182)
        dt_requested_291087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 59), 'dt_requested', False)
        # Processing the call keyword arguments (line 182)
        str_291088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'str', 'zoh')
        keyword_291089 = str_291088
        kwargs_291090 = {'method': keyword_291089}
        # Getting the type of 'c2d' (line 182)
        c2d_291082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'c2d', False)
        # Calling c2d(args, kwargs) (line 182)
        c2d_call_result_291091 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), c2d_291082, *[tuple_291083, dt_requested_291087], **kwargs_291090)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___291092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), c2d_call_result_291091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_291093 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___291092, int_291081)
        
        # Assigning a type to the variable 'tuple_var_assignment_289866' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289866', subscript_call_result_291093)
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_291094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to c2d(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_291096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'zeros_c' (line 182)
        zeros_c_291097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'zeros_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291096, zeros_c_291097)
        # Adding element type (line 182)
        # Getting the type of 'poles_c' (line 182)
        poles_c_291098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'poles_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291096, poles_c_291098)
        # Adding element type (line 182)
        # Getting the type of 'k_c' (line 182)
        k_c_291099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'k_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291096, k_c_291099)
        
        # Getting the type of 'dt_requested' (line 182)
        dt_requested_291100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 59), 'dt_requested', False)
        # Processing the call keyword arguments (line 182)
        str_291101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'str', 'zoh')
        keyword_291102 = str_291101
        kwargs_291103 = {'method': keyword_291102}
        # Getting the type of 'c2d' (line 182)
        c2d_291095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'c2d', False)
        # Calling c2d(args, kwargs) (line 182)
        c2d_call_result_291104 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), c2d_291095, *[tuple_291096, dt_requested_291100], **kwargs_291103)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___291105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), c2d_call_result_291104, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_291106 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___291105, int_291094)
        
        # Assigning a type to the variable 'tuple_var_assignment_289867' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289867', subscript_call_result_291106)
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_291107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to c2d(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_291109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'zeros_c' (line 182)
        zeros_c_291110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'zeros_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291109, zeros_c_291110)
        # Adding element type (line 182)
        # Getting the type of 'poles_c' (line 182)
        poles_c_291111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'poles_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291109, poles_c_291111)
        # Adding element type (line 182)
        # Getting the type of 'k_c' (line 182)
        k_c_291112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'k_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291109, k_c_291112)
        
        # Getting the type of 'dt_requested' (line 182)
        dt_requested_291113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 59), 'dt_requested', False)
        # Processing the call keyword arguments (line 182)
        str_291114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'str', 'zoh')
        keyword_291115 = str_291114
        kwargs_291116 = {'method': keyword_291115}
        # Getting the type of 'c2d' (line 182)
        c2d_291108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'c2d', False)
        # Calling c2d(args, kwargs) (line 182)
        c2d_call_result_291117 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), c2d_291108, *[tuple_291109, dt_requested_291113], **kwargs_291116)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___291118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), c2d_call_result_291117, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_291119 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___291118, int_291107)
        
        # Assigning a type to the variable 'tuple_var_assignment_289868' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289868', subscript_call_result_291119)
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_291120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 8), 'int')
        
        # Call to c2d(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'tuple' (line 182)
        tuple_291122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 182)
        # Adding element type (line 182)
        # Getting the type of 'zeros_c' (line 182)
        zeros_c_291123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'zeros_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291122, zeros_c_291123)
        # Adding element type (line 182)
        # Getting the type of 'poles_c' (line 182)
        poles_c_291124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'poles_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291122, poles_c_291124)
        # Adding element type (line 182)
        # Getting the type of 'k_c' (line 182)
        k_c_291125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 53), 'k_c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 35), tuple_291122, k_c_291125)
        
        # Getting the type of 'dt_requested' (line 182)
        dt_requested_291126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 59), 'dt_requested', False)
        # Processing the call keyword arguments (line 182)
        str_291127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 41), 'str', 'zoh')
        keyword_291128 = str_291127
        kwargs_291129 = {'method': keyword_291128}
        # Getting the type of 'c2d' (line 182)
        c2d_291121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'c2d', False)
        # Calling c2d(args, kwargs) (line 182)
        c2d_call_result_291130 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), c2d_291121, *[tuple_291122, dt_requested_291126], **kwargs_291129)
        
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___291131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), c2d_call_result_291130, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_291132 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), getitem___291131, int_291120)
        
        # Assigning a type to the variable 'tuple_var_assignment_289869' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289869', subscript_call_result_291132)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_289866' (line 182)
        tuple_var_assignment_289866_291133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289866')
        # Assigning a type to the variable 'zeros' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'zeros', tuple_var_assignment_289866_291133)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_289867' (line 182)
        tuple_var_assignment_289867_291134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289867')
        # Assigning a type to the variable 'poles' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'poles', tuple_var_assignment_289867_291134)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_289868' (line 182)
        tuple_var_assignment_289868_291135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289868')
        # Assigning a type to the variable 'k' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'k', tuple_var_assignment_289868_291135)
        
        # Assigning a Name to a Name (line 182):
        # Getting the type of 'tuple_var_assignment_289869' (line 182)
        tuple_var_assignment_289869_291136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'tuple_var_assignment_289869')
        # Assigning a type to the variable 'dt' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'dt', tuple_var_assignment_289869_291136)
        
        # Call to assert_array_almost_equal(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'zeros_d' (line 185)
        zeros_d_291138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 'zeros_d', False)
        # Getting the type of 'zeros' (line 185)
        zeros_291139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 43), 'zeros', False)
        # Processing the call keyword arguments (line 185)
        kwargs_291140 = {}
        # Getting the type of 'assert_array_almost_equal' (line 185)
        assert_array_almost_equal_291137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 185)
        assert_array_almost_equal_call_result_291141 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_array_almost_equal_291137, *[zeros_d_291138, zeros_291139], **kwargs_291140)
        
        
        # Call to assert_array_almost_equal(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'polls_d' (line 186)
        polls_d_291143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 34), 'polls_d', False)
        # Getting the type of 'poles' (line 186)
        poles_291144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 43), 'poles', False)
        # Processing the call keyword arguments (line 186)
        kwargs_291145 = {}
        # Getting the type of 'assert_array_almost_equal' (line 186)
        assert_array_almost_equal_291142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 186)
        assert_array_almost_equal_call_result_291146 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert_array_almost_equal_291142, *[polls_d_291143, poles_291144], **kwargs_291145)
        
        
        # Call to assert_almost_equal(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'k_d' (line 187)
        k_d_291148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'k_d', False)
        # Getting the type of 'k' (line 187)
        k_291149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'k', False)
        # Processing the call keyword arguments (line 187)
        kwargs_291150 = {}
        # Getting the type of 'assert_almost_equal' (line 187)
        assert_almost_equal_291147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 187)
        assert_almost_equal_call_result_291151 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), assert_almost_equal_291147, *[k_d_291148, k_291149], **kwargs_291150)
        
        
        # Call to assert_almost_equal(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'dt_requested' (line 188)
        dt_requested_291153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'dt_requested', False)
        # Getting the type of 'dt' (line 188)
        dt_291154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'dt', False)
        # Processing the call keyword arguments (line 188)
        kwargs_291155 = {}
        # Getting the type of 'assert_almost_equal' (line 188)
        assert_almost_equal_291152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 188)
        assert_almost_equal_call_result_291156 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), assert_almost_equal_291152, *[dt_requested_291153, dt_291154], **kwargs_291155)
        
        
        # ################# End of 'test_zerospolesgain(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zerospolesgain' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_291157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291157)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zerospolesgain'
        return stypy_return_type_291157


    @norecursion
    def test_gbt_with_sio_tf_and_zpk(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_gbt_with_sio_tf_and_zpk'
        module_type_store = module_type_store.open_function_context('test_gbt_with_sio_tf_and_zpk', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_gbt_with_sio_tf_and_zpk')
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_gbt_with_sio_tf_and_zpk.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_gbt_with_sio_tf_and_zpk', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_gbt_with_sio_tf_and_zpk', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_gbt_with_sio_tf_and_zpk(...)' code ##################

        str_291158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'str', "Test method='gbt' with alpha=0.25 for tf and zpk cases.")
        
        # Assigning a Num to a Name (line 193):
        
        # Assigning a Num to a Name (line 193):
        float_291159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 12), 'float')
        # Assigning a type to the variable 'A' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'A', float_291159)
        
        # Assigning a Num to a Name (line 194):
        
        # Assigning a Num to a Name (line 194):
        float_291160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'float')
        # Assigning a type to the variable 'B' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'B', float_291160)
        
        # Assigning a Num to a Name (line 195):
        
        # Assigning a Num to a Name (line 195):
        float_291161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 12), 'float')
        # Assigning a type to the variable 'C' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'C', float_291161)
        
        # Assigning a Num to a Name (line 196):
        
        # Assigning a Num to a Name (line 196):
        float_291162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 12), 'float')
        # Assigning a type to the variable 'D' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'D', float_291162)
        
        # Assigning a Call to a Tuple (line 199):
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_291163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        
        # Call to ss2tf(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'A' (line 199)
        A_291165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'A', False)
        # Getting the type of 'B' (line 199)
        B_291166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'B', False)
        # Getting the type of 'C' (line 199)
        C_291167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'C', False)
        # Getting the type of 'D' (line 199)
        D_291168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'D', False)
        # Processing the call keyword arguments (line 199)
        kwargs_291169 = {}
        # Getting the type of 'ss2tf' (line 199)
        ss2tf_291164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'ss2tf', False)
        # Calling ss2tf(args, kwargs) (line 199)
        ss2tf_call_result_291170 = invoke(stypy.reporting.localization.Localization(__file__, 199, 21), ss2tf_291164, *[A_291165, B_291166, C_291167, D_291168], **kwargs_291169)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___291171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), ss2tf_call_result_291170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_291172 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___291171, int_291163)
        
        # Assigning a type to the variable 'tuple_var_assignment_289870' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_289870', subscript_call_result_291172)
        
        # Assigning a Subscript to a Name (line 199):
        
        # Obtaining the type of the subscript
        int_291173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
        
        # Call to ss2tf(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'A' (line 199)
        A_291175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'A', False)
        # Getting the type of 'B' (line 199)
        B_291176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 30), 'B', False)
        # Getting the type of 'C' (line 199)
        C_291177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'C', False)
        # Getting the type of 'D' (line 199)
        D_291178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 36), 'D', False)
        # Processing the call keyword arguments (line 199)
        kwargs_291179 = {}
        # Getting the type of 'ss2tf' (line 199)
        ss2tf_291174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'ss2tf', False)
        # Calling ss2tf(args, kwargs) (line 199)
        ss2tf_call_result_291180 = invoke(stypy.reporting.localization.Localization(__file__, 199, 21), ss2tf_291174, *[A_291175, B_291176, C_291177, D_291178], **kwargs_291179)
        
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___291181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), ss2tf_call_result_291180, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_291182 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___291181, int_291173)
        
        # Assigning a type to the variable 'tuple_var_assignment_289871' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_289871', subscript_call_result_291182)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_289870' (line 199)
        tuple_var_assignment_289870_291183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_289870')
        # Assigning a type to the variable 'cnum' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'cnum', tuple_var_assignment_289870_291183)
        
        # Assigning a Name to a Name (line 199):
        # Getting the type of 'tuple_var_assignment_289871' (line 199)
        tuple_var_assignment_289871_291184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'tuple_var_assignment_289871')
        # Assigning a type to the variable 'cden' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'cden', tuple_var_assignment_289871_291184)
        
        # Assigning a Call to a Tuple (line 202):
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_291185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 8), 'int')
        
        # Call to ss2zpk(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'A' (line 202)
        A_291187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'A', False)
        # Getting the type of 'B' (line 202)
        B_291188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'B', False)
        # Getting the type of 'C' (line 202)
        C_291189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'C', False)
        # Getting the type of 'D' (line 202)
        D_291190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'D', False)
        # Processing the call keyword arguments (line 202)
        kwargs_291191 = {}
        # Getting the type of 'ss2zpk' (line 202)
        ss2zpk_291186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 202)
        ss2zpk_call_result_291192 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), ss2zpk_291186, *[A_291187, B_291188, C_291189, D_291190], **kwargs_291191)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___291193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ss2zpk_call_result_291192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_291194 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), getitem___291193, int_291185)
        
        # Assigning a type to the variable 'tuple_var_assignment_289872' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289872', subscript_call_result_291194)
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_291195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 8), 'int')
        
        # Call to ss2zpk(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'A' (line 202)
        A_291197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'A', False)
        # Getting the type of 'B' (line 202)
        B_291198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'B', False)
        # Getting the type of 'C' (line 202)
        C_291199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'C', False)
        # Getting the type of 'D' (line 202)
        D_291200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'D', False)
        # Processing the call keyword arguments (line 202)
        kwargs_291201 = {}
        # Getting the type of 'ss2zpk' (line 202)
        ss2zpk_291196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 202)
        ss2zpk_call_result_291202 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), ss2zpk_291196, *[A_291197, B_291198, C_291199, D_291200], **kwargs_291201)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___291203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ss2zpk_call_result_291202, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_291204 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), getitem___291203, int_291195)
        
        # Assigning a type to the variable 'tuple_var_assignment_289873' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289873', subscript_call_result_291204)
        
        # Assigning a Subscript to a Name (line 202):
        
        # Obtaining the type of the subscript
        int_291205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 8), 'int')
        
        # Call to ss2zpk(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'A' (line 202)
        A_291207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'A', False)
        # Getting the type of 'B' (line 202)
        B_291208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'B', False)
        # Getting the type of 'C' (line 202)
        C_291209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'C', False)
        # Getting the type of 'D' (line 202)
        D_291210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'D', False)
        # Processing the call keyword arguments (line 202)
        kwargs_291211 = {}
        # Getting the type of 'ss2zpk' (line 202)
        ss2zpk_291206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 202)
        ss2zpk_call_result_291212 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), ss2zpk_291206, *[A_291207, B_291208, C_291209, D_291210], **kwargs_291211)
        
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___291213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), ss2zpk_call_result_291212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_291214 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), getitem___291213, int_291205)
        
        # Assigning a type to the variable 'tuple_var_assignment_289874' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289874', subscript_call_result_291214)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_289872' (line 202)
        tuple_var_assignment_289872_291215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289872')
        # Assigning a type to the variable 'cz' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'cz', tuple_var_assignment_289872_291215)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_289873' (line 202)
        tuple_var_assignment_289873_291216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289873')
        # Assigning a type to the variable 'cp' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'cp', tuple_var_assignment_289873_291216)
        
        # Assigning a Name to a Name (line 202):
        # Getting the type of 'tuple_var_assignment_289874' (line 202)
        tuple_var_assignment_289874_291217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'tuple_var_assignment_289874')
        # Assigning a type to the variable 'ck' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'ck', tuple_var_assignment_289874_291217)
        
        # Assigning a Num to a Name (line 204):
        
        # Assigning a Num to a Name (line 204):
        float_291218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 12), 'float')
        # Assigning a type to the variable 'h' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'h', float_291218)
        
        # Assigning a Num to a Name (line 205):
        
        # Assigning a Num to a Name (line 205):
        float_291219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 16), 'float')
        # Assigning a type to the variable 'alpha' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'alpha', float_291219)
        
        # Assigning a BinOp to a Name (line 208):
        
        # Assigning a BinOp to a Name (line 208):
        int_291220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 14), 'int')
        int_291221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'int')
        # Getting the type of 'alpha' (line 208)
        alpha_291222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'alpha')
        # Applying the binary operator '-' (line 208)
        result_sub_291223 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 19), '-', int_291221, alpha_291222)
        
        # Getting the type of 'h' (line 208)
        h_291224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'h')
        # Applying the binary operator '*' (line 208)
        result_mul_291225 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 18), '*', result_sub_291223, h_291224)
        
        # Getting the type of 'A' (line 208)
        A_291226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 36), 'A')
        # Applying the binary operator '*' (line 208)
        result_mul_291227 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 34), '*', result_mul_291225, A_291226)
        
        # Applying the binary operator '+' (line 208)
        result_add_291228 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 14), '+', int_291220, result_mul_291227)
        
        int_291229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 42), 'int')
        # Getting the type of 'alpha' (line 208)
        alpha_291230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 46), 'alpha')
        # Getting the type of 'h' (line 208)
        h_291231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 54), 'h')
        # Applying the binary operator '*' (line 208)
        result_mul_291232 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 46), '*', alpha_291230, h_291231)
        
        # Getting the type of 'A' (line 208)
        A_291233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 58), 'A')
        # Applying the binary operator '*' (line 208)
        result_mul_291234 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 56), '*', result_mul_291232, A_291233)
        
        # Applying the binary operator '-' (line 208)
        result_sub_291235 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 42), '-', int_291229, result_mul_291234)
        
        # Applying the binary operator 'div' (line 208)
        result_div_291236 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 13), 'div', result_add_291228, result_sub_291235)
        
        # Assigning a type to the variable 'Ad' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'Ad', result_div_291236)
        
        # Assigning a BinOp to a Name (line 209):
        
        # Assigning a BinOp to a Name (line 209):
        # Getting the type of 'h' (line 209)
        h_291237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 13), 'h')
        # Getting the type of 'B' (line 209)
        B_291238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 17), 'B')
        # Applying the binary operator '*' (line 209)
        result_mul_291239 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 13), '*', h_291237, B_291238)
        
        int_291240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'int')
        # Getting the type of 'alpha' (line 209)
        alpha_291241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 26), 'alpha')
        # Getting the type of 'h' (line 209)
        h_291242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 34), 'h')
        # Applying the binary operator '*' (line 209)
        result_mul_291243 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 26), '*', alpha_291241, h_291242)
        
        # Getting the type of 'A' (line 209)
        A_291244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'A')
        # Applying the binary operator '*' (line 209)
        result_mul_291245 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 36), '*', result_mul_291243, A_291244)
        
        # Applying the binary operator '-' (line 209)
        result_sub_291246 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 22), '-', int_291240, result_mul_291245)
        
        # Applying the binary operator 'div' (line 209)
        result_div_291247 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 19), 'div', result_mul_291239, result_sub_291246)
        
        # Assigning a type to the variable 'Bd' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'Bd', result_div_291247)
        
        # Assigning a BinOp to a Name (line 210):
        
        # Assigning a BinOp to a Name (line 210):
        # Getting the type of 'C' (line 210)
        C_291248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'C')
        int_291249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'int')
        # Getting the type of 'alpha' (line 210)
        alpha_291250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'alpha')
        # Getting the type of 'h' (line 210)
        h_291251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'h')
        # Applying the binary operator '*' (line 210)
        result_mul_291252 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 22), '*', alpha_291250, h_291251)
        
        # Getting the type of 'A' (line 210)
        A_291253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'A')
        # Applying the binary operator '*' (line 210)
        result_mul_291254 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 32), '*', result_mul_291252, A_291253)
        
        # Applying the binary operator '-' (line 210)
        result_sub_291255 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 18), '-', int_291249, result_mul_291254)
        
        # Applying the binary operator 'div' (line 210)
        result_div_291256 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 13), 'div', C_291248, result_sub_291255)
        
        # Assigning a type to the variable 'Cd' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'Cd', result_div_291256)
        
        # Assigning a BinOp to a Name (line 211):
        
        # Assigning a BinOp to a Name (line 211):
        # Getting the type of 'D' (line 211)
        D_291257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'D')
        # Getting the type of 'alpha' (line 211)
        alpha_291258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'alpha')
        # Getting the type of 'C' (line 211)
        C_291259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'C')
        # Applying the binary operator '*' (line 211)
        result_mul_291260 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 17), '*', alpha_291258, C_291259)
        
        # Getting the type of 'Bd' (line 211)
        Bd_291261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'Bd')
        # Applying the binary operator '*' (line 211)
        result_mul_291262 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 27), '*', result_mul_291260, Bd_291261)
        
        # Applying the binary operator '+' (line 211)
        result_add_291263 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 13), '+', D_291257, result_mul_291262)
        
        # Assigning a type to the variable 'Dd' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'Dd', result_add_291263)
        
        # Assigning a Call to a Tuple (line 214):
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_291264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
        
        # Call to ss2tf(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'Ad' (line 214)
        Ad_291266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'Ad', False)
        # Getting the type of 'Bd' (line 214)
        Bd_291267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'Bd', False)
        # Getting the type of 'Cd' (line 214)
        Cd_291268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 35), 'Cd', False)
        # Getting the type of 'Dd' (line 214)
        Dd_291269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'Dd', False)
        # Processing the call keyword arguments (line 214)
        kwargs_291270 = {}
        # Getting the type of 'ss2tf' (line 214)
        ss2tf_291265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'ss2tf', False)
        # Calling ss2tf(args, kwargs) (line 214)
        ss2tf_call_result_291271 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), ss2tf_291265, *[Ad_291266, Bd_291267, Cd_291268, Dd_291269], **kwargs_291270)
        
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___291272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), ss2tf_call_result_291271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_291273 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___291272, int_291264)
        
        # Assigning a type to the variable 'tuple_var_assignment_289875' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_289875', subscript_call_result_291273)
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        int_291274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
        
        # Call to ss2tf(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'Ad' (line 214)
        Ad_291276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'Ad', False)
        # Getting the type of 'Bd' (line 214)
        Bd_291277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'Bd', False)
        # Getting the type of 'Cd' (line 214)
        Cd_291278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 35), 'Cd', False)
        # Getting the type of 'Dd' (line 214)
        Dd_291279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'Dd', False)
        # Processing the call keyword arguments (line 214)
        kwargs_291280 = {}
        # Getting the type of 'ss2tf' (line 214)
        ss2tf_291275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'ss2tf', False)
        # Calling ss2tf(args, kwargs) (line 214)
        ss2tf_call_result_291281 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), ss2tf_291275, *[Ad_291276, Bd_291277, Cd_291278, Dd_291279], **kwargs_291280)
        
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___291282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), ss2tf_call_result_291281, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_291283 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___291282, int_291274)
        
        # Assigning a type to the variable 'tuple_var_assignment_289876' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_289876', subscript_call_result_291283)
        
        # Assigning a Name to a Name (line 214):
        # Getting the type of 'tuple_var_assignment_289875' (line 214)
        tuple_var_assignment_289875_291284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_289875')
        # Assigning a type to the variable 'dnum' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'dnum', tuple_var_assignment_289875_291284)
        
        # Assigning a Name to a Name (line 214):
        # Getting the type of 'tuple_var_assignment_289876' (line 214)
        tuple_var_assignment_289876_291285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_289876')
        # Assigning a type to the variable 'dden' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 14), 'dden', tuple_var_assignment_289876_291285)
        
        # Assigning a Call to a Tuple (line 217):
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_291286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to c2d(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_291288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'cnum' (line 217)
        cnum_291289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'cnum', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291288, cnum_291289)
        # Adding element type (line 217)
        # Getting the type of 'cden' (line 217)
        cden_291290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 'cden', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291288, cden_291290)
        
        # Getting the type of 'h' (line 217)
        h_291291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'h', False)
        # Processing the call keyword arguments (line 217)
        str_291292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'str', 'gbt')
        keyword_291293 = str_291292
        # Getting the type of 'alpha' (line 217)
        alpha_291294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 70), 'alpha', False)
        keyword_291295 = alpha_291294
        kwargs_291296 = {'alpha': keyword_291295, 'method': keyword_291293}
        # Getting the type of 'c2d' (line 217)
        c2d_291287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 217)
        c2d_call_result_291297 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), c2d_291287, *[tuple_291288, h_291291], **kwargs_291296)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___291298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), c2d_call_result_291297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_291299 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___291298, int_291286)
        
        # Assigning a type to the variable 'tuple_var_assignment_289877' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289877', subscript_call_result_291299)
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_291300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to c2d(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_291302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'cnum' (line 217)
        cnum_291303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'cnum', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291302, cnum_291303)
        # Adding element type (line 217)
        # Getting the type of 'cden' (line 217)
        cden_291304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 'cden', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291302, cden_291304)
        
        # Getting the type of 'h' (line 217)
        h_291305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'h', False)
        # Processing the call keyword arguments (line 217)
        str_291306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'str', 'gbt')
        keyword_291307 = str_291306
        # Getting the type of 'alpha' (line 217)
        alpha_291308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 70), 'alpha', False)
        keyword_291309 = alpha_291308
        kwargs_291310 = {'alpha': keyword_291309, 'method': keyword_291307}
        # Getting the type of 'c2d' (line 217)
        c2d_291301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 217)
        c2d_call_result_291311 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), c2d_291301, *[tuple_291302, h_291305], **kwargs_291310)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___291312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), c2d_call_result_291311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_291313 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___291312, int_291300)
        
        # Assigning a type to the variable 'tuple_var_assignment_289878' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289878', subscript_call_result_291313)
        
        # Assigning a Subscript to a Name (line 217):
        
        # Obtaining the type of the subscript
        int_291314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 8), 'int')
        
        # Call to c2d(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_291316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        # Getting the type of 'cnum' (line 217)
        cnum_291317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'cnum', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291316, cnum_291317)
        # Adding element type (line 217)
        # Getting the type of 'cden' (line 217)
        cden_291318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 'cden', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 34), tuple_291316, cden_291318)
        
        # Getting the type of 'h' (line 217)
        h_291319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 47), 'h', False)
        # Processing the call keyword arguments (line 217)
        str_291320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 57), 'str', 'gbt')
        keyword_291321 = str_291320
        # Getting the type of 'alpha' (line 217)
        alpha_291322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 70), 'alpha', False)
        keyword_291323 = alpha_291322
        kwargs_291324 = {'alpha': keyword_291323, 'method': keyword_291321}
        # Getting the type of 'c2d' (line 217)
        c2d_291315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'c2d', False)
        # Calling c2d(args, kwargs) (line 217)
        c2d_call_result_291325 = invoke(stypy.reporting.localization.Localization(__file__, 217, 29), c2d_291315, *[tuple_291316, h_291319], **kwargs_291324)
        
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___291326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), c2d_call_result_291325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_291327 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), getitem___291326, int_291314)
        
        # Assigning a type to the variable 'tuple_var_assignment_289879' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289879', subscript_call_result_291327)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_289877' (line 217)
        tuple_var_assignment_289877_291328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289877')
        # Assigning a type to the variable 'c2dnum' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'c2dnum', tuple_var_assignment_289877_291328)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_289878' (line 217)
        tuple_var_assignment_289878_291329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289878')
        # Assigning a type to the variable 'c2dden' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'c2dden', tuple_var_assignment_289878_291329)
        
        # Assigning a Name to a Name (line 217):
        # Getting the type of 'tuple_var_assignment_289879' (line 217)
        tuple_var_assignment_289879_291330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'tuple_var_assignment_289879')
        # Assigning a type to the variable 'dt' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'dt', tuple_var_assignment_289879_291330)
        
        # Call to assert_allclose(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'dnum' (line 219)
        dnum_291332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'dnum', False)
        # Getting the type of 'c2dnum' (line 219)
        c2dnum_291333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'c2dnum', False)
        # Processing the call keyword arguments (line 219)
        kwargs_291334 = {}
        # Getting the type of 'assert_allclose' (line 219)
        assert_allclose_291331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 219)
        assert_allclose_call_result_291335 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), assert_allclose_291331, *[dnum_291332, c2dnum_291333], **kwargs_291334)
        
        
        # Call to assert_allclose(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'dden' (line 220)
        dden_291337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'dden', False)
        # Getting the type of 'c2dden' (line 220)
        c2dden_291338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'c2dden', False)
        # Processing the call keyword arguments (line 220)
        kwargs_291339 = {}
        # Getting the type of 'assert_allclose' (line 220)
        assert_allclose_291336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 220)
        assert_allclose_call_result_291340 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), assert_allclose_291336, *[dden_291337, c2dden_291338], **kwargs_291339)
        
        
        # Assigning a Call to a Tuple (line 223):
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_291341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to ss2zpk(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'Ad' (line 223)
        Ad_291343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'Ad', False)
        # Getting the type of 'Bd' (line 223)
        Bd_291344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'Bd', False)
        # Getting the type of 'Cd' (line 223)
        Cd_291345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 36), 'Cd', False)
        # Getting the type of 'Dd' (line 223)
        Dd_291346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'Dd', False)
        # Processing the call keyword arguments (line 223)
        kwargs_291347 = {}
        # Getting the type of 'ss2zpk' (line 223)
        ss2zpk_291342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 223)
        ss2zpk_call_result_291348 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), ss2zpk_291342, *[Ad_291343, Bd_291344, Cd_291345, Dd_291346], **kwargs_291347)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___291349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), ss2zpk_call_result_291348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_291350 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___291349, int_291341)
        
        # Assigning a type to the variable 'tuple_var_assignment_289880' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289880', subscript_call_result_291350)
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_291351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to ss2zpk(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'Ad' (line 223)
        Ad_291353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'Ad', False)
        # Getting the type of 'Bd' (line 223)
        Bd_291354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'Bd', False)
        # Getting the type of 'Cd' (line 223)
        Cd_291355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 36), 'Cd', False)
        # Getting the type of 'Dd' (line 223)
        Dd_291356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'Dd', False)
        # Processing the call keyword arguments (line 223)
        kwargs_291357 = {}
        # Getting the type of 'ss2zpk' (line 223)
        ss2zpk_291352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 223)
        ss2zpk_call_result_291358 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), ss2zpk_291352, *[Ad_291353, Bd_291354, Cd_291355, Dd_291356], **kwargs_291357)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___291359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), ss2zpk_call_result_291358, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_291360 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___291359, int_291351)
        
        # Assigning a type to the variable 'tuple_var_assignment_289881' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289881', subscript_call_result_291360)
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_291361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to ss2zpk(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'Ad' (line 223)
        Ad_291363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'Ad', False)
        # Getting the type of 'Bd' (line 223)
        Bd_291364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'Bd', False)
        # Getting the type of 'Cd' (line 223)
        Cd_291365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 36), 'Cd', False)
        # Getting the type of 'Dd' (line 223)
        Dd_291366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'Dd', False)
        # Processing the call keyword arguments (line 223)
        kwargs_291367 = {}
        # Getting the type of 'ss2zpk' (line 223)
        ss2zpk_291362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'ss2zpk', False)
        # Calling ss2zpk(args, kwargs) (line 223)
        ss2zpk_call_result_291368 = invoke(stypy.reporting.localization.Localization(__file__, 223, 21), ss2zpk_291362, *[Ad_291363, Bd_291364, Cd_291365, Dd_291366], **kwargs_291367)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___291369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), ss2zpk_call_result_291368, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_291370 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___291369, int_291361)
        
        # Assigning a type to the variable 'tuple_var_assignment_289882' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289882', subscript_call_result_291370)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_289880' (line 223)
        tuple_var_assignment_289880_291371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289880')
        # Assigning a type to the variable 'dz' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'dz', tuple_var_assignment_289880_291371)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_289881' (line 223)
        tuple_var_assignment_289881_291372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289881')
        # Assigning a type to the variable 'dp' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'dp', tuple_var_assignment_289881_291372)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_289882' (line 223)
        tuple_var_assignment_289882_291373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_289882')
        # Assigning a type to the variable 'dk' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'dk', tuple_var_assignment_289882_291373)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_291374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to c2d(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_291376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'cz' (line 226)
        cz_291377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'cz', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291376, cz_291377)
        # Adding element type (line 226)
        # Getting the type of 'cp' (line 226)
        cp_291378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'cp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291376, cp_291378)
        # Adding element type (line 226)
        # Getting the type of 'ck' (line 226)
        ck_291379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ck', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291376, ck_291379)
        
        # Getting the type of 'h' (line 226)
        h_291380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'h', False)
        # Processing the call keyword arguments (line 226)
        str_291381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 59), 'str', 'gbt')
        keyword_291382 = str_291381
        # Getting the type of 'alpha' (line 226)
        alpha_291383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 72), 'alpha', False)
        keyword_291384 = alpha_291383
        kwargs_291385 = {'alpha': keyword_291384, 'method': keyword_291382}
        # Getting the type of 'c2d' (line 226)
        c2d_291375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'c2d', False)
        # Calling c2d(args, kwargs) (line 226)
        c2d_call_result_291386 = invoke(stypy.reporting.localization.Localization(__file__, 226, 31), c2d_291375, *[tuple_291376, h_291380], **kwargs_291385)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___291387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), c2d_call_result_291386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_291388 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___291387, int_291374)
        
        # Assigning a type to the variable 'tuple_var_assignment_289883' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289883', subscript_call_result_291388)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_291389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to c2d(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_291391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'cz' (line 226)
        cz_291392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'cz', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291391, cz_291392)
        # Adding element type (line 226)
        # Getting the type of 'cp' (line 226)
        cp_291393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'cp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291391, cp_291393)
        # Adding element type (line 226)
        # Getting the type of 'ck' (line 226)
        ck_291394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ck', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291391, ck_291394)
        
        # Getting the type of 'h' (line 226)
        h_291395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'h', False)
        # Processing the call keyword arguments (line 226)
        str_291396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 59), 'str', 'gbt')
        keyword_291397 = str_291396
        # Getting the type of 'alpha' (line 226)
        alpha_291398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 72), 'alpha', False)
        keyword_291399 = alpha_291398
        kwargs_291400 = {'alpha': keyword_291399, 'method': keyword_291397}
        # Getting the type of 'c2d' (line 226)
        c2d_291390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'c2d', False)
        # Calling c2d(args, kwargs) (line 226)
        c2d_call_result_291401 = invoke(stypy.reporting.localization.Localization(__file__, 226, 31), c2d_291390, *[tuple_291391, h_291395], **kwargs_291400)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___291402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), c2d_call_result_291401, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_291403 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___291402, int_291389)
        
        # Assigning a type to the variable 'tuple_var_assignment_289884' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289884', subscript_call_result_291403)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_291404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to c2d(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_291406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'cz' (line 226)
        cz_291407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'cz', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291406, cz_291407)
        # Adding element type (line 226)
        # Getting the type of 'cp' (line 226)
        cp_291408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'cp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291406, cp_291408)
        # Adding element type (line 226)
        # Getting the type of 'ck' (line 226)
        ck_291409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ck', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291406, ck_291409)
        
        # Getting the type of 'h' (line 226)
        h_291410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'h', False)
        # Processing the call keyword arguments (line 226)
        str_291411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 59), 'str', 'gbt')
        keyword_291412 = str_291411
        # Getting the type of 'alpha' (line 226)
        alpha_291413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 72), 'alpha', False)
        keyword_291414 = alpha_291413
        kwargs_291415 = {'alpha': keyword_291414, 'method': keyword_291412}
        # Getting the type of 'c2d' (line 226)
        c2d_291405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'c2d', False)
        # Calling c2d(args, kwargs) (line 226)
        c2d_call_result_291416 = invoke(stypy.reporting.localization.Localization(__file__, 226, 31), c2d_291405, *[tuple_291406, h_291410], **kwargs_291415)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___291417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), c2d_call_result_291416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_291418 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___291417, int_291404)
        
        # Assigning a type to the variable 'tuple_var_assignment_289885' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289885', subscript_call_result_291418)
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_291419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to c2d(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_291421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'cz' (line 226)
        cz_291422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 36), 'cz', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291421, cz_291422)
        # Adding element type (line 226)
        # Getting the type of 'cp' (line 226)
        cp_291423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'cp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291421, cp_291423)
        # Adding element type (line 226)
        # Getting the type of 'ck' (line 226)
        ck_291424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 44), 'ck', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 36), tuple_291421, ck_291424)
        
        # Getting the type of 'h' (line 226)
        h_291425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 49), 'h', False)
        # Processing the call keyword arguments (line 226)
        str_291426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 59), 'str', 'gbt')
        keyword_291427 = str_291426
        # Getting the type of 'alpha' (line 226)
        alpha_291428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 72), 'alpha', False)
        keyword_291429 = alpha_291428
        kwargs_291430 = {'alpha': keyword_291429, 'method': keyword_291427}
        # Getting the type of 'c2d' (line 226)
        c2d_291420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'c2d', False)
        # Calling c2d(args, kwargs) (line 226)
        c2d_call_result_291431 = invoke(stypy.reporting.localization.Localization(__file__, 226, 31), c2d_291420, *[tuple_291421, h_291425], **kwargs_291430)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___291432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), c2d_call_result_291431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_291433 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___291432, int_291419)
        
        # Assigning a type to the variable 'tuple_var_assignment_289886' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289886', subscript_call_result_291433)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_289883' (line 226)
        tuple_var_assignment_289883_291434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289883')
        # Assigning a type to the variable 'c2dz' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'c2dz', tuple_var_assignment_289883_291434)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_289884' (line 226)
        tuple_var_assignment_289884_291435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289884')
        # Assigning a type to the variable 'c2dp' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 14), 'c2dp', tuple_var_assignment_289884_291435)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_289885' (line 226)
        tuple_var_assignment_289885_291436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289885')
        # Assigning a type to the variable 'c2dk' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'c2dk', tuple_var_assignment_289885_291436)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_289886' (line 226)
        tuple_var_assignment_289886_291437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_289886')
        # Assigning a type to the variable 'dt' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 26), 'dt', tuple_var_assignment_289886_291437)
        
        # Call to assert_allclose(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'dz' (line 228)
        dz_291439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'dz', False)
        # Getting the type of 'c2dz' (line 228)
        c2dz_291440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'c2dz', False)
        # Processing the call keyword arguments (line 228)
        kwargs_291441 = {}
        # Getting the type of 'assert_allclose' (line 228)
        assert_allclose_291438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 228)
        assert_allclose_call_result_291442 = invoke(stypy.reporting.localization.Localization(__file__, 228, 8), assert_allclose_291438, *[dz_291439, c2dz_291440], **kwargs_291441)
        
        
        # Call to assert_allclose(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'dp' (line 229)
        dp_291444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'dp', False)
        # Getting the type of 'c2dp' (line 229)
        c2dp_291445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'c2dp', False)
        # Processing the call keyword arguments (line 229)
        kwargs_291446 = {}
        # Getting the type of 'assert_allclose' (line 229)
        assert_allclose_291443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 229)
        assert_allclose_call_result_291447 = invoke(stypy.reporting.localization.Localization(__file__, 229, 8), assert_allclose_291443, *[dp_291444, c2dp_291445], **kwargs_291446)
        
        
        # Call to assert_allclose(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'dk' (line 230)
        dk_291449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'dk', False)
        # Getting the type of 'c2dk' (line 230)
        c2dk_291450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'c2dk', False)
        # Processing the call keyword arguments (line 230)
        kwargs_291451 = {}
        # Getting the type of 'assert_allclose' (line 230)
        assert_allclose_291448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 230)
        assert_allclose_call_result_291452 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), assert_allclose_291448, *[dk_291449, c2dk_291450], **kwargs_291451)
        
        
        # ################# End of 'test_gbt_with_sio_tf_and_zpk(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_gbt_with_sio_tf_and_zpk' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_291453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_gbt_with_sio_tf_and_zpk'
        return stypy_return_type_291453


    @norecursion
    def test_discrete_approx(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_discrete_approx'
        module_type_store = module_type_store.open_function_context('test_discrete_approx', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_discrete_approx')
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_discrete_approx.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_discrete_approx', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_discrete_approx', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_discrete_approx(...)' code ##################

        str_291454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, (-1)), 'str', '\n        Test that the solution to the discrete approximation of a continuous\n        system actually approximates the solution to the continuous system.\n        This is an indirect test of the correctness of the implementation\n        of cont2discrete.\n        ')

        @norecursion
        def u(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'u'
            module_type_store = module_type_store.open_function_context('u', 240, 8, False)
            
            # Passed parameters checking function
            u.stypy_localization = localization
            u.stypy_type_of_self = None
            u.stypy_type_store = module_type_store
            u.stypy_function_name = 'u'
            u.stypy_param_names_list = ['t']
            u.stypy_varargs_param_name = None
            u.stypy_kwargs_param_name = None
            u.stypy_call_defaults = defaults
            u.stypy_call_varargs = varargs
            u.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'u', ['t'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'u', localization, ['t'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'u(...)' code ##################

            
            # Call to sin(...): (line 241)
            # Processing the call arguments (line 241)
            float_291457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 26), 'float')
            # Getting the type of 't' (line 241)
            t_291458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 32), 't', False)
            # Applying the binary operator '*' (line 241)
            result_mul_291459 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 26), '*', float_291457, t_291458)
            
            # Processing the call keyword arguments (line 241)
            kwargs_291460 = {}
            # Getting the type of 'np' (line 241)
            np_291455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 19), 'np', False)
            # Obtaining the member 'sin' of a type (line 241)
            sin_291456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 19), np_291455, 'sin')
            # Calling sin(args, kwargs) (line 241)
            sin_call_result_291461 = invoke(stypy.reporting.localization.Localization(__file__, 241, 19), sin_291456, *[result_mul_291459], **kwargs_291460)
            
            # Assigning a type to the variable 'stypy_return_type' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'stypy_return_type', sin_call_result_291461)
            
            # ################# End of 'u(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'u' in the type store
            # Getting the type of 'stypy_return_type' (line 240)
            stypy_return_type_291462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_291462)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'u'
            return stypy_return_type_291462

        # Assigning a type to the variable 'u' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'u', u)
        
        # Assigning a Call to a Name (line 243):
        
        # Assigning a Call to a Name (line 243):
        
        # Call to array(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_291465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_291466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        float_291467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 22), list_291466, float_291467)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 21), list_291465, list_291466)
        
        # Processing the call keyword arguments (line 243)
        kwargs_291468 = {}
        # Getting the type of 'np' (line 243)
        np_291463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 243)
        array_291464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), np_291463, 'array')
        # Calling array(args, kwargs) (line 243)
        array_call_result_291469 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), array_291464, *[list_291465], **kwargs_291468)
        
        # Assigning a type to the variable 'a' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'a', array_call_result_291469)
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to array(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_291472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        
        # Obtaining an instance of the builtin type 'list' (line 244)
        list_291473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 244)
        # Adding element type (line 244)
        float_291474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 22), list_291473, float_291474)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 21), list_291472, list_291473)
        
        # Processing the call keyword arguments (line 244)
        kwargs_291475 = {}
        # Getting the type of 'np' (line 244)
        np_291470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 244)
        array_291471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), np_291470, 'array')
        # Calling array(args, kwargs) (line 244)
        array_call_result_291476 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), array_291471, *[list_291472], **kwargs_291475)
        
        # Assigning a type to the variable 'b' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'b', array_call_result_291476)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to array(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_291479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_291480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        float_291481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 22), list_291480, float_291481)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 21), list_291479, list_291480)
        
        # Processing the call keyword arguments (line 245)
        kwargs_291482 = {}
        # Getting the type of 'np' (line 245)
        np_291477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 245)
        array_291478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), np_291477, 'array')
        # Calling array(args, kwargs) (line 245)
        array_call_result_291483 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), array_291478, *[list_291479], **kwargs_291482)
        
        # Assigning a type to the variable 'c' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'c', array_call_result_291483)
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to array(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_291486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        
        # Obtaining an instance of the builtin type 'list' (line 246)
        list_291487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 246)
        # Adding element type (line 246)
        float_291488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 22), list_291487, float_291488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 21), list_291486, list_291487)
        
        # Processing the call keyword arguments (line 246)
        kwargs_291489 = {}
        # Getting the type of 'np' (line 246)
        np_291484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 246)
        array_291485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), np_291484, 'array')
        # Calling array(args, kwargs) (line 246)
        array_call_result_291490 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), array_291485, *[list_291486], **kwargs_291489)
        
        # Assigning a type to the variable 'd' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'd', array_call_result_291490)
        
        # Assigning a Num to a Name (line 247):
        
        # Assigning a Num to a Name (line 247):
        float_291491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 13), 'float')
        # Assigning a type to the variable 'x0' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'x0', float_291491)
        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to linspace(...): (line 249)
        # Processing the call arguments (line 249)
        int_291494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 24), 'int')
        float_291495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 27), 'float')
        int_291496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 33), 'int')
        # Processing the call keyword arguments (line 249)
        kwargs_291497 = {}
        # Getting the type of 'np' (line 249)
        np_291492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 249)
        linspace_291493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), np_291492, 'linspace')
        # Calling linspace(args, kwargs) (line 249)
        linspace_call_result_291498 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), linspace_291493, *[int_291494, float_291495, int_291496], **kwargs_291497)
        
        # Assigning a type to the variable 't' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 't', linspace_call_result_291498)
        
        # Assigning a BinOp to a Name (line 250):
        
        # Assigning a BinOp to a Name (line 250):
        
        # Obtaining the type of the subscript
        int_291499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 15), 'int')
        # Getting the type of 't' (line 250)
        t_291500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 13), 't')
        # Obtaining the member '__getitem__' of a type (line 250)
        getitem___291501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 13), t_291500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 250)
        subscript_call_result_291502 = invoke(stypy.reporting.localization.Localization(__file__, 250, 13), getitem___291501, int_291499)
        
        
        # Obtaining the type of the subscript
        int_291503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 22), 'int')
        # Getting the type of 't' (line 250)
        t_291504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 't')
        # Obtaining the member '__getitem__' of a type (line 250)
        getitem___291505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), t_291504, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 250)
        subscript_call_result_291506 = invoke(stypy.reporting.localization.Localization(__file__, 250, 20), getitem___291505, int_291503)
        
        # Applying the binary operator '-' (line 250)
        result_sub_291507 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 13), '-', subscript_call_result_291502, subscript_call_result_291506)
        
        # Assigning a type to the variable 'dt' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'dt', result_sub_291507)
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to u(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 't' (line 251)
        t_291509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 't', False)
        # Processing the call keyword arguments (line 251)
        kwargs_291510 = {}
        # Getting the type of 'u' (line 251)
        u_291508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'u', False)
        # Calling u(args, kwargs) (line 251)
        u_call_result_291511 = invoke(stypy.reporting.localization.Localization(__file__, 251, 13), u_291508, *[t_291509], **kwargs_291510)
        
        # Assigning a type to the variable 'u1' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'u1', u_call_result_291511)
        
        # Assigning a Call to a Tuple (line 254):
        
        # Assigning a Subscript to a Name (line 254):
        
        # Obtaining the type of the subscript
        int_291512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'int')
        
        # Call to lsim2(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_291514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'a' (line 254)
        a_291515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291514, a_291515)
        # Adding element type (line 254)
        # Getting the type of 'b' (line 254)
        b_291516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291514, b_291516)
        # Adding element type (line 254)
        # Getting the type of 'c' (line 254)
        c_291517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291514, c_291517)
        # Adding element type (line 254)
        # Getting the type of 'd' (line 254)
        d_291518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291514, d_291518)
        
        # Processing the call keyword arguments (line 254)
        # Getting the type of 't' (line 254)
        t_291519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 46), 't', False)
        keyword_291520 = t_291519
        # Getting the type of 'u1' (line 254)
        u1_291521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 51), 'u1', False)
        keyword_291522 = u1_291521
        # Getting the type of 'x0' (line 254)
        x0_291523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'x0', False)
        keyword_291524 = x0_291523
        float_291525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 35), 'float')
        keyword_291526 = float_291525
        float_291527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 46), 'float')
        keyword_291528 = float_291527
        kwargs_291529 = {'X0': keyword_291524, 'rtol': keyword_291526, 'U': keyword_291522, 'T': keyword_291520, 'atol': keyword_291528}
        # Getting the type of 'lsim2' (line 254)
        lsim2_291513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'lsim2', False)
        # Calling lsim2(args, kwargs) (line 254)
        lsim2_call_result_291530 = invoke(stypy.reporting.localization.Localization(__file__, 254, 24), lsim2_291513, *[tuple_291514], **kwargs_291529)
        
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___291531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), lsim2_call_result_291530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_291532 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), getitem___291531, int_291512)
        
        # Assigning a type to the variable 'tuple_var_assignment_289887' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289887', subscript_call_result_291532)
        
        # Assigning a Subscript to a Name (line 254):
        
        # Obtaining the type of the subscript
        int_291533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'int')
        
        # Call to lsim2(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_291535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'a' (line 254)
        a_291536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291535, a_291536)
        # Adding element type (line 254)
        # Getting the type of 'b' (line 254)
        b_291537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291535, b_291537)
        # Adding element type (line 254)
        # Getting the type of 'c' (line 254)
        c_291538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291535, c_291538)
        # Adding element type (line 254)
        # Getting the type of 'd' (line 254)
        d_291539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291535, d_291539)
        
        # Processing the call keyword arguments (line 254)
        # Getting the type of 't' (line 254)
        t_291540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 46), 't', False)
        keyword_291541 = t_291540
        # Getting the type of 'u1' (line 254)
        u1_291542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 51), 'u1', False)
        keyword_291543 = u1_291542
        # Getting the type of 'x0' (line 254)
        x0_291544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'x0', False)
        keyword_291545 = x0_291544
        float_291546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 35), 'float')
        keyword_291547 = float_291546
        float_291548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 46), 'float')
        keyword_291549 = float_291548
        kwargs_291550 = {'X0': keyword_291545, 'rtol': keyword_291547, 'U': keyword_291543, 'T': keyword_291541, 'atol': keyword_291549}
        # Getting the type of 'lsim2' (line 254)
        lsim2_291534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'lsim2', False)
        # Calling lsim2(args, kwargs) (line 254)
        lsim2_call_result_291551 = invoke(stypy.reporting.localization.Localization(__file__, 254, 24), lsim2_291534, *[tuple_291535], **kwargs_291550)
        
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___291552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), lsim2_call_result_291551, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_291553 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), getitem___291552, int_291533)
        
        # Assigning a type to the variable 'tuple_var_assignment_289888' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289888', subscript_call_result_291553)
        
        # Assigning a Subscript to a Name (line 254):
        
        # Obtaining the type of the subscript
        int_291554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 8), 'int')
        
        # Call to lsim2(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'tuple' (line 254)
        tuple_291556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 254)
        # Adding element type (line 254)
        # Getting the type of 'a' (line 254)
        a_291557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 31), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291556, a_291557)
        # Adding element type (line 254)
        # Getting the type of 'b' (line 254)
        b_291558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291556, b_291558)
        # Adding element type (line 254)
        # Getting the type of 'c' (line 254)
        c_291559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 37), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291556, c_291559)
        # Adding element type (line 254)
        # Getting the type of 'd' (line 254)
        d_291560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 40), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_291556, d_291560)
        
        # Processing the call keyword arguments (line 254)
        # Getting the type of 't' (line 254)
        t_291561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 46), 't', False)
        keyword_291562 = t_291561
        # Getting the type of 'u1' (line 254)
        u1_291563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 51), 'u1', False)
        keyword_291564 = u1_291563
        # Getting the type of 'x0' (line 254)
        x0_291565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'x0', False)
        keyword_291566 = x0_291565
        float_291567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 35), 'float')
        keyword_291568 = float_291567
        float_291569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 46), 'float')
        keyword_291570 = float_291569
        kwargs_291571 = {'X0': keyword_291566, 'rtol': keyword_291568, 'U': keyword_291564, 'T': keyword_291562, 'atol': keyword_291570}
        # Getting the type of 'lsim2' (line 254)
        lsim2_291555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'lsim2', False)
        # Calling lsim2(args, kwargs) (line 254)
        lsim2_call_result_291572 = invoke(stypy.reporting.localization.Localization(__file__, 254, 24), lsim2_291555, *[tuple_291556], **kwargs_291571)
        
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___291573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 8), lsim2_call_result_291572, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_291574 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), getitem___291573, int_291554)
        
        # Assigning a type to the variable 'tuple_var_assignment_289889' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289889', subscript_call_result_291574)
        
        # Assigning a Name to a Name (line 254):
        # Getting the type of 'tuple_var_assignment_289887' (line 254)
        tuple_var_assignment_289887_291575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289887')
        # Assigning a type to the variable 't' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 't', tuple_var_assignment_289887_291575)
        
        # Assigning a Name to a Name (line 254):
        # Getting the type of 'tuple_var_assignment_289888' (line 254)
        tuple_var_assignment_289888_291576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289888')
        # Assigning a type to the variable 'yout' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'yout', tuple_var_assignment_289888_291576)
        
        # Assigning a Name to a Name (line 254):
        # Getting the type of 'tuple_var_assignment_289889' (line 254)
        tuple_var_assignment_289889_291577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'tuple_var_assignment_289889')
        # Assigning a type to the variable 'xout' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'xout', tuple_var_assignment_289889_291577)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to c2d(...): (line 258)
        # Processing the call arguments (line 258)
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_291579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        # Getting the type of 'a' (line 258)
        a_291580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), tuple_291579, a_291580)
        # Adding element type (line 258)
        # Getting the type of 'b' (line 258)
        b_291581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), tuple_291579, b_291581)
        # Adding element type (line 258)
        # Getting the type of 'c' (line 258)
        c_291582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 26), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), tuple_291579, c_291582)
        # Adding element type (line 258)
        # Getting the type of 'd' (line 258)
        d_291583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 20), tuple_291579, d_291583)
        
        # Getting the type of 'dt' (line 258)
        dt_291584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'dt', False)
        # Processing the call keyword arguments (line 258)
        str_291585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 44), 'str', 'bilinear')
        keyword_291586 = str_291585
        kwargs_291587 = {'method': keyword_291586}
        # Getting the type of 'c2d' (line 258)
        c2d_291578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'c2d', False)
        # Calling c2d(args, kwargs) (line 258)
        c2d_call_result_291588 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), c2d_291578, *[tuple_291579, dt_291584], **kwargs_291587)
        
        # Assigning a type to the variable 'dsys' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'dsys', c2d_call_result_291588)
        
        # Assigning a BinOp to a Name (line 262):
        
        # Assigning a BinOp to a Name (line 262):
        float_291589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 13), 'float')
        
        # Obtaining the type of the subscript
        int_291590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 24), 'int')
        slice_291591 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 262, 20), None, int_291590, None)
        # Getting the type of 'u1' (line 262)
        u1_291592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'u1')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___291593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), u1_291592, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_291594 = invoke(stypy.reporting.localization.Localization(__file__, 262, 20), getitem___291593, slice_291591)
        
        
        # Obtaining the type of the subscript
        int_291595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
        slice_291596 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 262, 30), int_291595, None, None)
        # Getting the type of 'u1' (line 262)
        u1_291597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'u1')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___291598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), u1_291597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_291599 = invoke(stypy.reporting.localization.Localization(__file__, 262, 30), getitem___291598, slice_291596)
        
        # Applying the binary operator '+' (line 262)
        result_add_291600 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 20), '+', subscript_call_result_291594, subscript_call_result_291599)
        
        # Applying the binary operator '*' (line 262)
        result_mul_291601 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 13), '*', float_291589, result_add_291600)
        
        # Assigning a type to the variable 'u2' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'u2', result_mul_291601)
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_291602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
        slice_291603 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 13), None, int_291602, None)
        # Getting the type of 't' (line 263)
        t_291604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 13), 't')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___291605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 13), t_291604, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_291606 = invoke(stypy.reporting.localization.Localization(__file__, 263, 13), getitem___291605, slice_291603)
        
        # Assigning a type to the variable 't2' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 't2', subscript_call_result_291606)
        
        # Assigning a Call to a Tuple (line 264):
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_291607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'int')
        
        # Call to dlsim(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'dsys' (line 264)
        dsys_291609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'dsys', False)
        # Processing the call keyword arguments (line 264)
        
        # Call to reshape(...): (line 264)
        # Processing the call arguments (line 264)
        int_291612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 49), 'int')
        int_291613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 53), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_291614 = {}
        # Getting the type of 'u2' (line 264)
        u2_291610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 38), 'u2', False)
        # Obtaining the member 'reshape' of a type (line 264)
        reshape_291611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 38), u2_291610, 'reshape')
        # Calling reshape(args, kwargs) (line 264)
        reshape_call_result_291615 = invoke(stypy.reporting.localization.Localization(__file__, 264, 38), reshape_291611, *[int_291612, int_291613], **kwargs_291614)
        
        keyword_291616 = reshape_call_result_291615
        # Getting the type of 't2' (line 264)
        t2_291617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 59), 't2', False)
        keyword_291618 = t2_291617
        # Getting the type of 'x0' (line 264)
        x0_291619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'x0', False)
        keyword_291620 = x0_291619
        kwargs_291621 = {'x0': keyword_291620, 'u': keyword_291616, 't': keyword_291618}
        # Getting the type of 'dlsim' (line 264)
        dlsim_291608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 264)
        dlsim_call_result_291622 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), dlsim_291608, *[dsys_291609], **kwargs_291621)
        
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___291623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), dlsim_call_result_291622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_291624 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___291623, int_291607)
        
        # Assigning a type to the variable 'tuple_var_assignment_289890' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289890', subscript_call_result_291624)
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_291625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'int')
        
        # Call to dlsim(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'dsys' (line 264)
        dsys_291627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'dsys', False)
        # Processing the call keyword arguments (line 264)
        
        # Call to reshape(...): (line 264)
        # Processing the call arguments (line 264)
        int_291630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 49), 'int')
        int_291631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 53), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_291632 = {}
        # Getting the type of 'u2' (line 264)
        u2_291628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 38), 'u2', False)
        # Obtaining the member 'reshape' of a type (line 264)
        reshape_291629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 38), u2_291628, 'reshape')
        # Calling reshape(args, kwargs) (line 264)
        reshape_call_result_291633 = invoke(stypy.reporting.localization.Localization(__file__, 264, 38), reshape_291629, *[int_291630, int_291631], **kwargs_291632)
        
        keyword_291634 = reshape_call_result_291633
        # Getting the type of 't2' (line 264)
        t2_291635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 59), 't2', False)
        keyword_291636 = t2_291635
        # Getting the type of 'x0' (line 264)
        x0_291637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'x0', False)
        keyword_291638 = x0_291637
        kwargs_291639 = {'x0': keyword_291638, 'u': keyword_291634, 't': keyword_291636}
        # Getting the type of 'dlsim' (line 264)
        dlsim_291626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 264)
        dlsim_call_result_291640 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), dlsim_291626, *[dsys_291627], **kwargs_291639)
        
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___291641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), dlsim_call_result_291640, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_291642 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___291641, int_291625)
        
        # Assigning a type to the variable 'tuple_var_assignment_289891' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289891', subscript_call_result_291642)
        
        # Assigning a Subscript to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_291643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 8), 'int')
        
        # Call to dlsim(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'dsys' (line 264)
        dsys_291645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'dsys', False)
        # Processing the call keyword arguments (line 264)
        
        # Call to reshape(...): (line 264)
        # Processing the call arguments (line 264)
        int_291648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 49), 'int')
        int_291649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 53), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_291650 = {}
        # Getting the type of 'u2' (line 264)
        u2_291646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 38), 'u2', False)
        # Obtaining the member 'reshape' of a type (line 264)
        reshape_291647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 38), u2_291646, 'reshape')
        # Calling reshape(args, kwargs) (line 264)
        reshape_call_result_291651 = invoke(stypy.reporting.localization.Localization(__file__, 264, 38), reshape_291647, *[int_291648, int_291649], **kwargs_291650)
        
        keyword_291652 = reshape_call_result_291651
        # Getting the type of 't2' (line 264)
        t2_291653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 59), 't2', False)
        keyword_291654 = t2_291653
        # Getting the type of 'x0' (line 264)
        x0_291655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'x0', False)
        keyword_291656 = x0_291655
        kwargs_291657 = {'x0': keyword_291656, 'u': keyword_291652, 't': keyword_291654}
        # Getting the type of 'dlsim' (line 264)
        dlsim_291644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'dlsim', False)
        # Calling dlsim(args, kwargs) (line 264)
        dlsim_call_result_291658 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), dlsim_291644, *[dsys_291645], **kwargs_291657)
        
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___291659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), dlsim_call_result_291658, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_291660 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), getitem___291659, int_291643)
        
        # Assigning a type to the variable 'tuple_var_assignment_289892' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289892', subscript_call_result_291660)
        
        # Assigning a Name to a Name (line 264):
        # Getting the type of 'tuple_var_assignment_289890' (line 264)
        tuple_var_assignment_289890_291661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289890')
        # Assigning a type to the variable 'td2' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'td2', tuple_var_assignment_289890_291661)
        
        # Assigning a Name to a Name (line 264):
        # Getting the type of 'tuple_var_assignment_289891' (line 264)
        tuple_var_assignment_289891_291662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289891')
        # Assigning a type to the variable 'yd2' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 13), 'yd2', tuple_var_assignment_289891_291662)
        
        # Assigning a Name to a Name (line 264):
        # Getting the type of 'tuple_var_assignment_289892' (line 264)
        tuple_var_assignment_289892_291663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'tuple_var_assignment_289892')
        # Assigning a type to the variable 'xd2' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 18), 'xd2', tuple_var_assignment_289892_291663)
        
        # Assigning a BinOp to a Name (line 269):
        
        # Assigning a BinOp to a Name (line 269):
        float_291664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 15), 'float')
        
        # Obtaining the type of the subscript
        int_291665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 28), 'int')
        slice_291666 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 22), None, int_291665, None)
        # Getting the type of 'yout' (line 269)
        yout_291667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'yout')
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___291668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), yout_291667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_291669 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), getitem___291668, slice_291666)
        
        
        # Obtaining the type of the subscript
        int_291670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 39), 'int')
        slice_291671 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 34), int_291670, None, None)
        # Getting the type of 'yout' (line 269)
        yout_291672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 34), 'yout')
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___291673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 34), yout_291672, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_291674 = invoke(stypy.reporting.localization.Localization(__file__, 269, 34), getitem___291673, slice_291671)
        
        # Applying the binary operator '+' (line 269)
        result_add_291675 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 22), '+', subscript_call_result_291669, subscript_call_result_291674)
        
        # Applying the binary operator '*' (line 269)
        result_mul_291676 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), '*', float_291664, result_add_291675)
        
        # Assigning a type to the variable 'ymid' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'ymid', result_mul_291676)
        
        # Call to assert_allclose(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Call to ravel(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_291680 = {}
        # Getting the type of 'yd2' (line 271)
        yd2_291678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 24), 'yd2', False)
        # Obtaining the member 'ravel' of a type (line 271)
        ravel_291679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 24), yd2_291678, 'ravel')
        # Calling ravel(args, kwargs) (line 271)
        ravel_call_result_291681 = invoke(stypy.reporting.localization.Localization(__file__, 271, 24), ravel_291679, *[], **kwargs_291680)
        
        # Getting the type of 'ymid' (line 271)
        ymid_291682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 'ymid', False)
        # Processing the call keyword arguments (line 271)
        float_291683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 48), 'float')
        keyword_291684 = float_291683
        kwargs_291685 = {'rtol': keyword_291684}
        # Getting the type of 'assert_allclose' (line 271)
        assert_allclose_291677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 271)
        assert_allclose_call_result_291686 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), assert_allclose_291677, *[ravel_call_result_291681, ymid_291682], **kwargs_291685)
        
        
        # ################# End of 'test_discrete_approx(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_discrete_approx' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_291687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_discrete_approx'
        return stypy_return_type_291687


    @norecursion
    def test_simo_tf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_simo_tf'
        module_type_store = module_type_store.open_function_context('test_simo_tf', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_simo_tf')
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_simo_tf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_simo_tf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_simo_tf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_simo_tf(...)' code ##################

        
        # Assigning a Tuple to a Name (line 275):
        
        # Assigning a Tuple to a Name (line 275):
        
        # Obtaining an instance of the builtin type 'tuple' (line 275)
        tuple_291688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 275)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_291689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_291690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_291691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), list_291690, int_291691)
        # Adding element type (line 275)
        int_291692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 15), list_291690, int_291692)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 14), list_291689, list_291690)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_291693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_291694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 23), list_291693, int_291694)
        # Adding element type (line 275)
        int_291695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 23), list_291693, int_291695)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 14), list_291689, list_291693)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 14), tuple_291688, list_291689)
        # Adding element type (line 275)
        
        # Obtaining an instance of the builtin type 'list' (line 275)
        list_291696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 275)
        # Adding element type (line 275)
        int_291697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 32), list_291696, int_291697)
        # Adding element type (line 275)
        int_291698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 32), list_291696, int_291698)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 14), tuple_291688, list_291696)
        
        # Assigning a type to the variable 'tf' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'tf', tuple_291688)
        
        # Assigning a Call to a Tuple (line 276):
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        int_291699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'int')
        
        # Call to c2d(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'tf' (line 276)
        tf_291701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'tf', False)
        float_291702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'float')
        # Processing the call keyword arguments (line 276)
        kwargs_291703 = {}
        # Getting the type of 'c2d' (line 276)
        c2d_291700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 276)
        c2d_call_result_291704 = invoke(stypy.reporting.localization.Localization(__file__, 276, 23), c2d_291700, *[tf_291701, float_291702], **kwargs_291703)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___291705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), c2d_call_result_291704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_291706 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___291705, int_291699)
        
        # Assigning a type to the variable 'tuple_var_assignment_289893' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289893', subscript_call_result_291706)
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        int_291707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'int')
        
        # Call to c2d(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'tf' (line 276)
        tf_291709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'tf', False)
        float_291710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'float')
        # Processing the call keyword arguments (line 276)
        kwargs_291711 = {}
        # Getting the type of 'c2d' (line 276)
        c2d_291708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 276)
        c2d_call_result_291712 = invoke(stypy.reporting.localization.Localization(__file__, 276, 23), c2d_291708, *[tf_291709, float_291710], **kwargs_291711)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___291713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), c2d_call_result_291712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_291714 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___291713, int_291707)
        
        # Assigning a type to the variable 'tuple_var_assignment_289894' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289894', subscript_call_result_291714)
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        int_291715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'int')
        
        # Call to c2d(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'tf' (line 276)
        tf_291717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'tf', False)
        float_291718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 31), 'float')
        # Processing the call keyword arguments (line 276)
        kwargs_291719 = {}
        # Getting the type of 'c2d' (line 276)
        c2d_291716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 276)
        c2d_call_result_291720 = invoke(stypy.reporting.localization.Localization(__file__, 276, 23), c2d_291716, *[tf_291717, float_291718], **kwargs_291719)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___291721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), c2d_call_result_291720, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_291722 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___291721, int_291715)
        
        # Assigning a type to the variable 'tuple_var_assignment_289895' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289895', subscript_call_result_291722)
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'tuple_var_assignment_289893' (line 276)
        tuple_var_assignment_289893_291723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289893')
        # Assigning a type to the variable 'num' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'num', tuple_var_assignment_289893_291723)
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'tuple_var_assignment_289894' (line 276)
        tuple_var_assignment_289894_291724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289894')
        # Assigning a type to the variable 'den' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 13), 'den', tuple_var_assignment_289894_291724)
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'tuple_var_assignment_289895' (line 276)
        tuple_var_assignment_289895_291725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_289895')
        # Assigning a type to the variable 'dt' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 'dt', tuple_var_assignment_289895_291725)
        
        # Call to assert_equal(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'dt' (line 278)
        dt_291727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'dt', False)
        float_291728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 25), 'float')
        # Processing the call keyword arguments (line 278)
        kwargs_291729 = {}
        # Getting the type of 'assert_equal' (line 278)
        assert_equal_291726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 278)
        assert_equal_call_result_291730 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), assert_equal_291726, *[dt_291727, float_291728], **kwargs_291729)
        
        
        # Call to assert_allclose(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'den' (line 279)
        den_291732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'den', False)
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_291733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        # Adding element type (line 279)
        int_291734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 29), list_291733, int_291734)
        # Adding element type (line 279)
        float_291735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 29), list_291733, float_291735)
        
        # Processing the call keyword arguments (line 279)
        float_291736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 53), 'float')
        keyword_291737 = float_291736
        kwargs_291738 = {'rtol': keyword_291737}
        # Getting the type of 'assert_allclose' (line 279)
        assert_allclose_291731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 279)
        assert_allclose_call_result_291739 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert_allclose_291731, *[den_291732, list_291733], **kwargs_291738)
        
        
        # Call to assert_allclose(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'num' (line 280)
        num_291741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'num', False)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_291742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_291743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_291744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 30), list_291743, int_291744)
        # Adding element type (line 280)
        int_291745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 30), list_291743, int_291745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 29), list_291742, list_291743)
        # Adding element type (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_291746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_291747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 39), list_291746, int_291747)
        # Adding element type (line 280)
        float_291748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 39), list_291746, float_291748)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 29), list_291742, list_291746)
        
        # Processing the call keyword arguments (line 280)
        float_291749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 63), 'float')
        keyword_291750 = float_291749
        kwargs_291751 = {'rtol': keyword_291750}
        # Getting the type of 'assert_allclose' (line 280)
        assert_allclose_291740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 280)
        assert_allclose_call_result_291752 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), assert_allclose_291740, *[num_291741, list_291742], **kwargs_291751)
        
        
        # ################# End of 'test_simo_tf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_simo_tf' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_291753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291753)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_simo_tf'
        return stypy_return_type_291753


    @norecursion
    def test_multioutput(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_multioutput'
        module_type_store = module_type_store.open_function_context('test_multioutput', 282, 4, False)
        # Assigning a type to the variable 'self' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_localization', localization)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_function_name', 'TestC2D.test_multioutput')
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2D.test_multioutput.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.test_multioutput', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_multioutput', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_multioutput(...)' code ##################

        
        # Assigning a Num to a Name (line 283):
        
        # Assigning a Num to a Name (line 283):
        float_291754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 13), 'float')
        # Assigning a type to the variable 'ts' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'ts', float_291754)
        
        # Assigning a Tuple to a Name (line 285):
        
        # Assigning a Tuple to a Name (line 285):
        
        # Obtaining an instance of the builtin type 'tuple' (line 285)
        tuple_291755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 14), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 285)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_291756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_291757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        int_291758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), list_291757, int_291758)
        # Adding element type (line 285)
        int_291759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 15), list_291757, int_291759)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 14), list_291756, list_291757)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_291760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        int_291761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), list_291760, int_291761)
        # Adding element type (line 285)
        int_291762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 24), list_291760, int_291762)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 14), list_291756, list_291760)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 14), tuple_291755, list_291756)
        # Adding element type (line 285)
        
        # Obtaining an instance of the builtin type 'list' (line 285)
        list_291763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 285)
        # Adding element type (line 285)
        int_291764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 33), list_291763, int_291764)
        # Adding element type (line 285)
        int_291765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 33), list_291763, int_291765)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 14), tuple_291755, list_291763)
        
        # Assigning a type to the variable 'tf' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'tf', tuple_291755)
        
        # Assigning a Call to a Tuple (line 286):
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_291766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        
        # Call to c2d(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'tf' (line 286)
        tf_291768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'tf', False)
        # Getting the type of 'ts' (line 286)
        ts_291769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'ts', False)
        # Processing the call keyword arguments (line 286)
        kwargs_291770 = {}
        # Getting the type of 'c2d' (line 286)
        c2d_291767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 286)
        c2d_call_result_291771 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), c2d_291767, *[tf_291768, ts_291769], **kwargs_291770)
        
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___291772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), c2d_call_result_291771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_291773 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___291772, int_291766)
        
        # Assigning a type to the variable 'tuple_var_assignment_289896' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289896', subscript_call_result_291773)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_291774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        
        # Call to c2d(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'tf' (line 286)
        tf_291776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'tf', False)
        # Getting the type of 'ts' (line 286)
        ts_291777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'ts', False)
        # Processing the call keyword arguments (line 286)
        kwargs_291778 = {}
        # Getting the type of 'c2d' (line 286)
        c2d_291775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 286)
        c2d_call_result_291779 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), c2d_291775, *[tf_291776, ts_291777], **kwargs_291778)
        
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___291780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), c2d_call_result_291779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_291781 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___291780, int_291774)
        
        # Assigning a type to the variable 'tuple_var_assignment_289897' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289897', subscript_call_result_291781)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_291782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        
        # Call to c2d(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'tf' (line 286)
        tf_291784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'tf', False)
        # Getting the type of 'ts' (line 286)
        ts_291785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'ts', False)
        # Processing the call keyword arguments (line 286)
        kwargs_291786 = {}
        # Getting the type of 'c2d' (line 286)
        c2d_291783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 23), 'c2d', False)
        # Calling c2d(args, kwargs) (line 286)
        c2d_call_result_291787 = invoke(stypy.reporting.localization.Localization(__file__, 286, 23), c2d_291783, *[tf_291784, ts_291785], **kwargs_291786)
        
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___291788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), c2d_call_result_291787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_291789 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___291788, int_291782)
        
        # Assigning a type to the variable 'tuple_var_assignment_289898' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289898', subscript_call_result_291789)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_289896' (line 286)
        tuple_var_assignment_289896_291790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289896')
        # Assigning a type to the variable 'num' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'num', tuple_var_assignment_289896_291790)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_289897' (line 286)
        tuple_var_assignment_289897_291791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289897')
        # Assigning a type to the variable 'den' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 13), 'den', tuple_var_assignment_289897_291791)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_289898' (line 286)
        tuple_var_assignment_289898_291792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_289898')
        # Assigning a type to the variable 'dt' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'dt', tuple_var_assignment_289898_291792)
        
        # Assigning a Tuple to a Name (line 288):
        
        # Assigning a Tuple to a Name (line 288):
        
        # Obtaining an instance of the builtin type 'tuple' (line 288)
        tuple_291793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 288)
        # Adding element type (line 288)
        
        # Obtaining the type of the subscript
        int_291794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'int')
        
        # Obtaining the type of the subscript
        int_291795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 18), 'int')
        # Getting the type of 'tf' (line 288)
        tf_291796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'tf')
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___291797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), tf_291796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_291798 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), getitem___291797, int_291795)
        
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___291799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), subscript_call_result_291798, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_291800 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), getitem___291799, int_291794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_291793, subscript_call_result_291800)
        # Adding element type (line 288)
        
        # Obtaining the type of the subscript
        int_291801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 28), 'int')
        # Getting the type of 'tf' (line 288)
        tf_291802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'tf')
        # Obtaining the member '__getitem__' of a type (line 288)
        getitem___291803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 25), tf_291802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 288)
        subscript_call_result_291804 = invoke(stypy.reporting.localization.Localization(__file__, 288, 25), getitem___291803, int_291801)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 15), tuple_291793, subscript_call_result_291804)
        
        # Assigning a type to the variable 'tf1' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'tf1', tuple_291793)
        
        # Assigning a Call to a Tuple (line 289):
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        int_291805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 8), 'int')
        
        # Call to c2d(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'tf1' (line 289)
        tf1_291807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'tf1', False)
        # Getting the type of 'ts' (line 289)
        ts_291808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'ts', False)
        # Processing the call keyword arguments (line 289)
        kwargs_291809 = {}
        # Getting the type of 'c2d' (line 289)
        c2d_291806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 289)
        c2d_call_result_291810 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), c2d_291806, *[tf1_291807, ts_291808], **kwargs_291809)
        
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___291811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), c2d_call_result_291810, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_291812 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), getitem___291811, int_291805)
        
        # Assigning a type to the variable 'tuple_var_assignment_289899' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289899', subscript_call_result_291812)
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        int_291813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 8), 'int')
        
        # Call to c2d(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'tf1' (line 289)
        tf1_291815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'tf1', False)
        # Getting the type of 'ts' (line 289)
        ts_291816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'ts', False)
        # Processing the call keyword arguments (line 289)
        kwargs_291817 = {}
        # Getting the type of 'c2d' (line 289)
        c2d_291814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 289)
        c2d_call_result_291818 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), c2d_291814, *[tf1_291815, ts_291816], **kwargs_291817)
        
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___291819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), c2d_call_result_291818, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_291820 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), getitem___291819, int_291813)
        
        # Assigning a type to the variable 'tuple_var_assignment_289900' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289900', subscript_call_result_291820)
        
        # Assigning a Subscript to a Name (line 289):
        
        # Obtaining the type of the subscript
        int_291821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 8), 'int')
        
        # Call to c2d(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'tf1' (line 289)
        tf1_291823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 30), 'tf1', False)
        # Getting the type of 'ts' (line 289)
        ts_291824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'ts', False)
        # Processing the call keyword arguments (line 289)
        kwargs_291825 = {}
        # Getting the type of 'c2d' (line 289)
        c2d_291822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 289)
        c2d_call_result_291826 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), c2d_291822, *[tf1_291823, ts_291824], **kwargs_291825)
        
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___291827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), c2d_call_result_291826, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_291828 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), getitem___291827, int_291821)
        
        # Assigning a type to the variable 'tuple_var_assignment_289901' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289901', subscript_call_result_291828)
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'tuple_var_assignment_289899' (line 289)
        tuple_var_assignment_289899_291829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289899')
        # Assigning a type to the variable 'num1' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'num1', tuple_var_assignment_289899_291829)
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'tuple_var_assignment_289900' (line 289)
        tuple_var_assignment_289900_291830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289900')
        # Assigning a type to the variable 'den1' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 14), 'den1', tuple_var_assignment_289900_291830)
        
        # Assigning a Name to a Name (line 289):
        # Getting the type of 'tuple_var_assignment_289901' (line 289)
        tuple_var_assignment_289901_291831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'tuple_var_assignment_289901')
        # Assigning a type to the variable 'dt1' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'dt1', tuple_var_assignment_289901_291831)
        
        # Assigning a Tuple to a Name (line 291):
        
        # Assigning a Tuple to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_291832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        
        # Obtaining the type of the subscript
        int_291833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 21), 'int')
        
        # Obtaining the type of the subscript
        int_291834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'int')
        # Getting the type of 'tf' (line 291)
        tf_291835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'tf')
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___291836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), tf_291835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_291837 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), getitem___291836, int_291834)
        
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___291838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), subscript_call_result_291837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_291839 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), getitem___291838, int_291833)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_291832, subscript_call_result_291839)
        # Adding element type (line 291)
        
        # Obtaining the type of the subscript
        int_291840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 28), 'int')
        # Getting the type of 'tf' (line 291)
        tf_291841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 25), 'tf')
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___291842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 25), tf_291841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_291843 = invoke(stypy.reporting.localization.Localization(__file__, 291, 25), getitem___291842, int_291840)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_291832, subscript_call_result_291843)
        
        # Assigning a type to the variable 'tf2' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tf2', tuple_291832)
        
        # Assigning a Call to a Tuple (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_291844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        
        # Call to c2d(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'tf2' (line 292)
        tf2_291846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'tf2', False)
        # Getting the type of 'ts' (line 292)
        ts_291847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 35), 'ts', False)
        # Processing the call keyword arguments (line 292)
        kwargs_291848 = {}
        # Getting the type of 'c2d' (line 292)
        c2d_291845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 292)
        c2d_call_result_291849 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), c2d_291845, *[tf2_291846, ts_291847], **kwargs_291848)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___291850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), c2d_call_result_291849, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_291851 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___291850, int_291844)
        
        # Assigning a type to the variable 'tuple_var_assignment_289902' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289902', subscript_call_result_291851)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_291852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        
        # Call to c2d(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'tf2' (line 292)
        tf2_291854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'tf2', False)
        # Getting the type of 'ts' (line 292)
        ts_291855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 35), 'ts', False)
        # Processing the call keyword arguments (line 292)
        kwargs_291856 = {}
        # Getting the type of 'c2d' (line 292)
        c2d_291853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 292)
        c2d_call_result_291857 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), c2d_291853, *[tf2_291854, ts_291855], **kwargs_291856)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___291858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), c2d_call_result_291857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_291859 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___291858, int_291852)
        
        # Assigning a type to the variable 'tuple_var_assignment_289903' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289903', subscript_call_result_291859)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_291860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        
        # Call to c2d(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'tf2' (line 292)
        tf2_291862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'tf2', False)
        # Getting the type of 'ts' (line 292)
        ts_291863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 35), 'ts', False)
        # Processing the call keyword arguments (line 292)
        kwargs_291864 = {}
        # Getting the type of 'c2d' (line 292)
        c2d_291861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'c2d', False)
        # Calling c2d(args, kwargs) (line 292)
        c2d_call_result_291865 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), c2d_291861, *[tf2_291862, ts_291863], **kwargs_291864)
        
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___291866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), c2d_call_result_291865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_291867 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___291866, int_291860)
        
        # Assigning a type to the variable 'tuple_var_assignment_289904' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289904', subscript_call_result_291867)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_289902' (line 292)
        tuple_var_assignment_289902_291868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289902')
        # Assigning a type to the variable 'num2' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'num2', tuple_var_assignment_289902_291868)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_289903' (line 292)
        tuple_var_assignment_289903_291869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289903')
        # Assigning a type to the variable 'den2' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'den2', tuple_var_assignment_289903_291869)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_289904' (line 292)
        tuple_var_assignment_289904_291870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_289904')
        # Assigning a type to the variable 'dt2' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'dt2', tuple_var_assignment_289904_291870)
        
        # Call to assert_equal(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'dt' (line 295)
        dt_291872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'dt', False)
        # Getting the type of 'dt1' (line 295)
        dt1_291873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'dt1', False)
        # Processing the call keyword arguments (line 295)
        kwargs_291874 = {}
        # Getting the type of 'assert_equal' (line 295)
        assert_equal_291871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 295)
        assert_equal_call_result_291875 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), assert_equal_291871, *[dt_291872, dt1_291873], **kwargs_291874)
        
        
        # Call to assert_equal(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'dt' (line 296)
        dt_291877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'dt', False)
        # Getting the type of 'dt2' (line 296)
        dt2_291878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 25), 'dt2', False)
        # Processing the call keyword arguments (line 296)
        kwargs_291879 = {}
        # Getting the type of 'assert_equal' (line 296)
        assert_equal_291876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 296)
        assert_equal_call_result_291880 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert_equal_291876, *[dt_291877, dt2_291878], **kwargs_291879)
        
        
        # Call to assert_allclose(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'num' (line 299)
        num_291882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'num', False)
        
        # Call to vstack(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Obtaining an instance of the builtin type 'tuple' (line 299)
        tuple_291885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 299)
        # Adding element type (line 299)
        # Getting the type of 'num1' (line 299)
        num1_291886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 40), 'num1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 40), tuple_291885, num1_291886)
        # Adding element type (line 299)
        # Getting the type of 'num2' (line 299)
        num2_291887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 46), 'num2', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 40), tuple_291885, num2_291887)
        
        # Processing the call keyword arguments (line 299)
        kwargs_291888 = {}
        # Getting the type of 'np' (line 299)
        np_291883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'np', False)
        # Obtaining the member 'vstack' of a type (line 299)
        vstack_291884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 29), np_291883, 'vstack')
        # Calling vstack(args, kwargs) (line 299)
        vstack_call_result_291889 = invoke(stypy.reporting.localization.Localization(__file__, 299, 29), vstack_291884, *[tuple_291885], **kwargs_291888)
        
        # Processing the call keyword arguments (line 299)
        float_291890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 59), 'float')
        keyword_291891 = float_291890
        kwargs_291892 = {'rtol': keyword_291891}
        # Getting the type of 'assert_allclose' (line 299)
        assert_allclose_291881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 299)
        assert_allclose_call_result_291893 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), assert_allclose_291881, *[num_291882, vstack_call_result_291889], **kwargs_291892)
        
        
        # Call to assert_allclose(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'den' (line 303)
        den_291895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'den', False)
        # Getting the type of 'den1' (line 303)
        den1_291896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 'den1', False)
        # Processing the call keyword arguments (line 303)
        float_291897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 40), 'float')
        keyword_291898 = float_291897
        kwargs_291899 = {'rtol': keyword_291898}
        # Getting the type of 'assert_allclose' (line 303)
        assert_allclose_291894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 303)
        assert_allclose_call_result_291900 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assert_allclose_291894, *[den_291895, den1_291896], **kwargs_291899)
        
        
        # Call to assert_allclose(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'den' (line 304)
        den_291902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'den', False)
        # Getting the type of 'den2' (line 304)
        den2_291903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'den2', False)
        # Processing the call keyword arguments (line 304)
        float_291904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 40), 'float')
        keyword_291905 = float_291904
        kwargs_291906 = {'rtol': keyword_291905}
        # Getting the type of 'assert_allclose' (line 304)
        assert_allclose_291901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 304)
        assert_allclose_call_result_291907 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assert_allclose_291901, *[den_291902, den2_291903], **kwargs_291906)
        
        
        # ################# End of 'test_multioutput(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_multioutput' in the type store
        # Getting the type of 'stypy_return_type' (line 282)
        stypy_return_type_291908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291908)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_multioutput'
        return stypy_return_type_291908


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2D.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestC2D' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'TestC2D', TestC2D)
# Declaration of the 'TestC2dLti' class

class TestC2dLti(object, ):

    @norecursion
    def test_c2d_ss(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_c2d_ss'
        module_type_store = module_type_store.open_function_context('test_c2d_ss', 307, 4, False)
        # Assigning a type to the variable 'self' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_localization', localization)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_function_name', 'TestC2dLti.test_c2d_ss')
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.test_c2d_ss', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_c2d_ss', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_c2d_ss(...)' code ##################

        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to array(...): (line 309)
        # Processing the call arguments (line 309)
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_291911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_291912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        float_291913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 22), list_291912, float_291913)
        # Adding element type (line 309)
        float_291914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 22), list_291912, float_291914)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 21), list_291911, list_291912)
        # Adding element type (line 309)
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_291915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        float_291916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), list_291915, float_291916)
        # Adding element type (line 309)
        float_291917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 35), list_291915, float_291917)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 21), list_291911, list_291915)
        
        # Processing the call keyword arguments (line 309)
        kwargs_291918 = {}
        # Getting the type of 'np' (line 309)
        np_291909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 309)
        array_291910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), np_291909, 'array')
        # Calling array(args, kwargs) (line 309)
        array_call_result_291919 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), array_291910, *[list_291911], **kwargs_291918)
        
        # Assigning a type to the variable 'A' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'A', array_call_result_291919)
        
        # Assigning a Call to a Name (line 310):
        
        # Assigning a Call to a Name (line 310):
        
        # Call to array(...): (line 310)
        # Processing the call arguments (line 310)
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_291922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_291923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        int_291924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 22), list_291923, int_291924)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 21), list_291922, list_291923)
        # Adding element type (line 310)
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_291925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        # Adding element type (line 310)
        int_291926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 27), list_291925, int_291926)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 21), list_291922, list_291925)
        
        # Processing the call keyword arguments (line 310)
        kwargs_291927 = {}
        # Getting the type of 'np' (line 310)
        np_291920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 310)
        array_291921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 12), np_291920, 'array')
        # Calling array(args, kwargs) (line 310)
        array_call_result_291928 = invoke(stypy.reporting.localization.Localization(__file__, 310, 12), array_291921, *[list_291922], **kwargs_291927)
        
        # Assigning a type to the variable 'B' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'B', array_call_result_291928)
        
        # Assigning a Call to a Name (line 311):
        
        # Assigning a Call to a Name (line 311):
        
        # Call to array(...): (line 311)
        # Processing the call arguments (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_291931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_291932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        int_291933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 22), list_291932, int_291933)
        # Adding element type (line 311)
        int_291934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 22), list_291932, int_291934)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 21), list_291931, list_291932)
        
        # Processing the call keyword arguments (line 311)
        kwargs_291935 = {}
        # Getting the type of 'np' (line 311)
        np_291929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 311)
        array_291930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), np_291929, 'array')
        # Calling array(args, kwargs) (line 311)
        array_call_result_291936 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), array_291930, *[list_291931], **kwargs_291935)
        
        # Assigning a type to the variable 'C' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'C', array_call_result_291936)
        
        # Assigning a Num to a Name (line 312):
        
        # Assigning a Num to a Name (line 312):
        int_291937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 12), 'int')
        # Assigning a type to the variable 'D' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'D', int_291937)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to array(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_291940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        
        # Obtaining an instance of the builtin type 'list' (line 314)
        list_291941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 314)
        # Adding element type (line 314)
        float_291942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 26), list_291941, float_291942)
        # Adding element type (line 314)
        float_291943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 26), list_291941, float_291943)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 25), list_291940, list_291941)
        # Adding element type (line 314)
        
        # Obtaining an instance of the builtin type 'list' (line 315)
        list_291944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 315)
        # Adding element type (line 315)
        float_291945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 26), list_291944, float_291945)
        # Adding element type (line 315)
        float_291946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 26), list_291944, float_291946)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 25), list_291940, list_291944)
        
        # Processing the call keyword arguments (line 314)
        kwargs_291947 = {}
        # Getting the type of 'np' (line 314)
        np_291938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 314)
        array_291939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), np_291938, 'array')
        # Calling array(args, kwargs) (line 314)
        array_call_result_291948 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), array_291939, *[list_291940], **kwargs_291947)
        
        # Assigning a type to the variable 'A_res' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'A_res', array_call_result_291948)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to array(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_291951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_291952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        float_291953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 26), list_291952, float_291953)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 25), list_291951, list_291952)
        # Adding element type (line 316)
        
        # Obtaining an instance of the builtin type 'list' (line 316)
        list_291954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 316)
        # Adding element type (line 316)
        float_291955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 47), list_291954, float_291955)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 25), list_291951, list_291954)
        
        # Processing the call keyword arguments (line 316)
        kwargs_291956 = {}
        # Getting the type of 'np' (line 316)
        np_291949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 316)
        array_291950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 16), np_291949, 'array')
        # Calling array(args, kwargs) (line 316)
        array_call_result_291957 = invoke(stypy.reporting.localization.Localization(__file__, 316, 16), array_291950, *[list_291951], **kwargs_291956)
        
        # Assigning a type to the variable 'B_res' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'B_res', array_call_result_291957)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to lti(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'A' (line 318)
        A_291959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'A', False)
        # Getting the type of 'B' (line 318)
        B_291960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'B', False)
        # Getting the type of 'C' (line 318)
        C_291961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'C', False)
        # Getting the type of 'D' (line 318)
        D_291962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 31), 'D', False)
        # Processing the call keyword arguments (line 318)
        kwargs_291963 = {}
        # Getting the type of 'lti' (line 318)
        lti_291958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'lti', False)
        # Calling lti(args, kwargs) (line 318)
        lti_call_result_291964 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), lti_291958, *[A_291959, B_291960, C_291961, D_291962], **kwargs_291963)
        
        # Assigning a type to the variable 'sys_ssc' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'sys_ssc', lti_call_result_291964)
        
        # Assigning a Call to a Name (line 319):
        
        # Assigning a Call to a Name (line 319):
        
        # Call to to_discrete(...): (line 319)
        # Processing the call arguments (line 319)
        float_291967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 38), 'float')
        # Processing the call keyword arguments (line 319)
        kwargs_291968 = {}
        # Getting the type of 'sys_ssc' (line 319)
        sys_ssc_291965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 18), 'sys_ssc', False)
        # Obtaining the member 'to_discrete' of a type (line 319)
        to_discrete_291966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 18), sys_ssc_291965, 'to_discrete')
        # Calling to_discrete(args, kwargs) (line 319)
        to_discrete_call_result_291969 = invoke(stypy.reporting.localization.Localization(__file__, 319, 18), to_discrete_291966, *[float_291967], **kwargs_291968)
        
        # Assigning a type to the variable 'sys_ssd' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'sys_ssd', to_discrete_call_result_291969)
        
        # Call to assert_allclose(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'sys_ssd' (line 321)
        sys_ssd_291971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 24), 'sys_ssd', False)
        # Obtaining the member 'A' of a type (line 321)
        A_291972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 24), sys_ssd_291971, 'A')
        # Getting the type of 'A_res' (line 321)
        A_res_291973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 35), 'A_res', False)
        # Processing the call keyword arguments (line 321)
        kwargs_291974 = {}
        # Getting the type of 'assert_allclose' (line 321)
        assert_allclose_291970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 321)
        assert_allclose_call_result_291975 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assert_allclose_291970, *[A_291972, A_res_291973], **kwargs_291974)
        
        
        # Call to assert_allclose(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'sys_ssd' (line 322)
        sys_ssd_291977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 24), 'sys_ssd', False)
        # Obtaining the member 'B' of a type (line 322)
        B_291978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 24), sys_ssd_291977, 'B')
        # Getting the type of 'B_res' (line 322)
        B_res_291979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 35), 'B_res', False)
        # Processing the call keyword arguments (line 322)
        kwargs_291980 = {}
        # Getting the type of 'assert_allclose' (line 322)
        assert_allclose_291976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 322)
        assert_allclose_call_result_291981 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), assert_allclose_291976, *[B_291978, B_res_291979], **kwargs_291980)
        
        
        # Call to assert_allclose(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'sys_ssd' (line 323)
        sys_ssd_291983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 24), 'sys_ssd', False)
        # Obtaining the member 'C' of a type (line 323)
        C_291984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 24), sys_ssd_291983, 'C')
        # Getting the type of 'C' (line 323)
        C_291985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 35), 'C', False)
        # Processing the call keyword arguments (line 323)
        kwargs_291986 = {}
        # Getting the type of 'assert_allclose' (line 323)
        assert_allclose_291982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 323)
        assert_allclose_call_result_291987 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), assert_allclose_291982, *[C_291984, C_291985], **kwargs_291986)
        
        
        # Call to assert_allclose(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'sys_ssd' (line 324)
        sys_ssd_291989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'sys_ssd', False)
        # Obtaining the member 'D' of a type (line 324)
        D_291990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 24), sys_ssd_291989, 'D')
        # Getting the type of 'D' (line 324)
        D_291991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 35), 'D', False)
        # Processing the call keyword arguments (line 324)
        kwargs_291992 = {}
        # Getting the type of 'assert_allclose' (line 324)
        assert_allclose_291988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 324)
        assert_allclose_call_result_291993 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), assert_allclose_291988, *[D_291990, D_291991], **kwargs_291992)
        
        
        # ################# End of 'test_c2d_ss(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_c2d_ss' in the type store
        # Getting the type of 'stypy_return_type' (line 307)
        stypy_return_type_291994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_291994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_c2d_ss'
        return stypy_return_type_291994


    @norecursion
    def test_c2d_tf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_c2d_tf'
        module_type_store = module_type_store.open_function_context('test_c2d_tf', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_localization', localization)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_function_name', 'TestC2dLti.test_c2d_tf')
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.test_c2d_tf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_c2d_tf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_c2d_tf(...)' code ##################

        
        # Assigning a Call to a Name (line 328):
        
        # Assigning a Call to a Name (line 328):
        
        # Call to lti(...): (line 328)
        # Processing the call arguments (line 328)
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_291996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        # Adding element type (line 328)
        float_291997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 18), list_291996, float_291997)
        # Adding element type (line 328)
        float_291998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 18), list_291996, float_291998)
        
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_291999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        # Adding element type (line 328)
        float_292000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 30), list_291999, float_292000)
        # Adding element type (line 328)
        float_292001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 30), list_291999, float_292001)
        
        # Processing the call keyword arguments (line 328)
        kwargs_292002 = {}
        # Getting the type of 'lti' (line 328)
        lti_291995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 14), 'lti', False)
        # Calling lti(args, kwargs) (line 328)
        lti_call_result_292003 = invoke(stypy.reporting.localization.Localization(__file__, 328, 14), lti_291995, *[list_291996, list_291999], **kwargs_292002)
        
        # Assigning a type to the variable 'sys' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'sys', lti_call_result_292003)
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to to_discrete(...): (line 329)
        # Processing the call arguments (line 329)
        float_292006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 30), 'float')
        # Processing the call keyword arguments (line 329)
        kwargs_292007 = {}
        # Getting the type of 'sys' (line 329)
        sys_292004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 14), 'sys', False)
        # Obtaining the member 'to_discrete' of a type (line 329)
        to_discrete_292005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 14), sys_292004, 'to_discrete')
        # Calling to_discrete(args, kwargs) (line 329)
        to_discrete_call_result_292008 = invoke(stypy.reporting.localization.Localization(__file__, 329, 14), to_discrete_292005, *[float_292006], **kwargs_292007)
        
        # Assigning a type to the variable 'sys' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'sys', to_discrete_call_result_292008)
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to array(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Obtaining an instance of the builtin type 'list' (line 332)
        list_292011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 332)
        # Adding element type (line 332)
        float_292012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 27), list_292011, float_292012)
        # Adding element type (line 332)
        float_292013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 27), list_292011, float_292013)
        
        # Processing the call keyword arguments (line 332)
        kwargs_292014 = {}
        # Getting the type of 'np' (line 332)
        np_292009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 332)
        array_292010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 18), np_292009, 'array')
        # Calling array(args, kwargs) (line 332)
        array_call_result_292015 = invoke(stypy.reporting.localization.Localization(__file__, 332, 18), array_292010, *[list_292011], **kwargs_292014)
        
        # Assigning a type to the variable 'num_res' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'num_res', array_call_result_292015)
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to array(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Obtaining an instance of the builtin type 'list' (line 333)
        list_292018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 333)
        # Adding element type (line 333)
        float_292019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 27), list_292018, float_292019)
        # Adding element type (line 333)
        float_292020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 27), list_292018, float_292020)
        
        # Processing the call keyword arguments (line 333)
        kwargs_292021 = {}
        # Getting the type of 'np' (line 333)
        np_292016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 333)
        array_292017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 18), np_292016, 'array')
        # Calling array(args, kwargs) (line 333)
        array_call_result_292022 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), array_292017, *[list_292018], **kwargs_292021)
        
        # Assigning a type to the variable 'den_res' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'den_res', array_call_result_292022)
        
        # Call to assert_allclose(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'sys' (line 336)
        sys_292024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'sys', False)
        # Obtaining the member 'den' of a type (line 336)
        den_292025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 24), sys_292024, 'den')
        # Getting the type of 'den_res' (line 336)
        den_res_292026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 33), 'den_res', False)
        # Processing the call keyword arguments (line 336)
        float_292027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 47), 'float')
        keyword_292028 = float_292027
        kwargs_292029 = {'atol': keyword_292028}
        # Getting the type of 'assert_allclose' (line 336)
        assert_allclose_292023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 336)
        assert_allclose_call_result_292030 = invoke(stypy.reporting.localization.Localization(__file__, 336, 8), assert_allclose_292023, *[den_292025, den_res_292026], **kwargs_292029)
        
        
        # Call to assert_allclose(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'sys' (line 337)
        sys_292032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'sys', False)
        # Obtaining the member 'num' of a type (line 337)
        num_292033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), sys_292032, 'num')
        # Getting the type of 'num_res' (line 337)
        num_res_292034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 33), 'num_res', False)
        # Processing the call keyword arguments (line 337)
        float_292035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 47), 'float')
        keyword_292036 = float_292035
        kwargs_292037 = {'atol': keyword_292036}
        # Getting the type of 'assert_allclose' (line 337)
        assert_allclose_292031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 337)
        assert_allclose_call_result_292038 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), assert_allclose_292031, *[num_292033, num_res_292034], **kwargs_292037)
        
        
        # ################# End of 'test_c2d_tf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_c2d_tf' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_292039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_c2d_tf'
        return stypy_return_type_292039


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 306, 0, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestC2dLti' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'TestC2dLti', TestC2dLti)
# Declaration of the 'TestC2dLti' class

class TestC2dLti(object, ):

    @norecursion
    def test_c2d_ss(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_c2d_ss'
        module_type_store = module_type_store.open_function_context('test_c2d_ss', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_localization', localization)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_function_name', 'TestC2dLti.test_c2d_ss')
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2dLti.test_c2d_ss.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.test_c2d_ss', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_c2d_ss', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_c2d_ss(...)' code ##################

        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to array(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Obtaining an instance of the builtin type 'list' (line 342)
        list_292042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 342)
        # Adding element type (line 342)
        
        # Obtaining an instance of the builtin type 'list' (line 342)
        list_292043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 342)
        # Adding element type (line 342)
        float_292044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 22), list_292043, float_292044)
        # Adding element type (line 342)
        float_292045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 22), list_292043, float_292045)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 21), list_292042, list_292043)
        # Adding element type (line 342)
        
        # Obtaining an instance of the builtin type 'list' (line 342)
        list_292046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 342)
        # Adding element type (line 342)
        float_292047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 35), list_292046, float_292047)
        # Adding element type (line 342)
        float_292048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 35), list_292046, float_292048)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 21), list_292042, list_292046)
        
        # Processing the call keyword arguments (line 342)
        kwargs_292049 = {}
        # Getting the type of 'np' (line 342)
        np_292040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 342)
        array_292041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), np_292040, 'array')
        # Calling array(args, kwargs) (line 342)
        array_call_result_292050 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), array_292041, *[list_292042], **kwargs_292049)
        
        # Assigning a type to the variable 'A' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'A', array_call_result_292050)
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to array(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_292053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_292054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_292055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 22), list_292054, int_292055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_292053, list_292054)
        # Adding element type (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_292056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_292057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_292056, int_292057)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 21), list_292053, list_292056)
        
        # Processing the call keyword arguments (line 343)
        kwargs_292058 = {}
        # Getting the type of 'np' (line 343)
        np_292051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 343)
        array_292052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), np_292051, 'array')
        # Calling array(args, kwargs) (line 343)
        array_call_result_292059 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), array_292052, *[list_292053], **kwargs_292058)
        
        # Assigning a type to the variable 'B' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'B', array_call_result_292059)
        
        # Assigning a Call to a Name (line 344):
        
        # Assigning a Call to a Name (line 344):
        
        # Call to array(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_292062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_292063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_292064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), list_292063, int_292064)
        # Adding element type (line 344)
        int_292065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 22), list_292063, int_292065)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 21), list_292062, list_292063)
        
        # Processing the call keyword arguments (line 344)
        kwargs_292066 = {}
        # Getting the type of 'np' (line 344)
        np_292060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 344)
        array_292061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), np_292060, 'array')
        # Calling array(args, kwargs) (line 344)
        array_call_result_292067 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), array_292061, *[list_292062], **kwargs_292066)
        
        # Assigning a type to the variable 'C' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'C', array_call_result_292067)
        
        # Assigning a Num to a Name (line 345):
        
        # Assigning a Num to a Name (line 345):
        int_292068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'int')
        # Assigning a type to the variable 'D' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'D', int_292068)
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to array(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_292071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        # Adding element type (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 347)
        list_292072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 347)
        # Adding element type (line 347)
        float_292073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 26), list_292072, float_292073)
        # Adding element type (line 347)
        float_292074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 26), list_292072, float_292074)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), list_292071, list_292072)
        # Adding element type (line 347)
        
        # Obtaining an instance of the builtin type 'list' (line 348)
        list_292075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 348)
        # Adding element type (line 348)
        float_292076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 26), list_292075, float_292076)
        # Adding element type (line 348)
        float_292077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 46), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 26), list_292075, float_292077)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 25), list_292071, list_292075)
        
        # Processing the call keyword arguments (line 347)
        kwargs_292078 = {}
        # Getting the type of 'np' (line 347)
        np_292069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 347)
        array_292070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), np_292069, 'array')
        # Calling array(args, kwargs) (line 347)
        array_call_result_292079 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), array_292070, *[list_292071], **kwargs_292078)
        
        # Assigning a type to the variable 'A_res' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'A_res', array_call_result_292079)
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to array(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_292082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_292083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        float_292084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 26), list_292083, float_292084)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 25), list_292082, list_292083)
        # Adding element type (line 349)
        
        # Obtaining an instance of the builtin type 'list' (line 349)
        list_292085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 349)
        # Adding element type (line 349)
        float_292086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 47), list_292085, float_292086)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 25), list_292082, list_292085)
        
        # Processing the call keyword arguments (line 349)
        kwargs_292087 = {}
        # Getting the type of 'np' (line 349)
        np_292080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 349)
        array_292081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 16), np_292080, 'array')
        # Calling array(args, kwargs) (line 349)
        array_call_result_292088 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), array_292081, *[list_292082], **kwargs_292087)
        
        # Assigning a type to the variable 'B_res' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'B_res', array_call_result_292088)
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to lti(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'A' (line 351)
        A_292090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 22), 'A', False)
        # Getting the type of 'B' (line 351)
        B_292091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 25), 'B', False)
        # Getting the type of 'C' (line 351)
        C_292092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 28), 'C', False)
        # Getting the type of 'D' (line 351)
        D_292093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'D', False)
        # Processing the call keyword arguments (line 351)
        kwargs_292094 = {}
        # Getting the type of 'lti' (line 351)
        lti_292089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'lti', False)
        # Calling lti(args, kwargs) (line 351)
        lti_call_result_292095 = invoke(stypy.reporting.localization.Localization(__file__, 351, 18), lti_292089, *[A_292090, B_292091, C_292092, D_292093], **kwargs_292094)
        
        # Assigning a type to the variable 'sys_ssc' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'sys_ssc', lti_call_result_292095)
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to to_discrete(...): (line 352)
        # Processing the call arguments (line 352)
        float_292098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'float')
        # Processing the call keyword arguments (line 352)
        kwargs_292099 = {}
        # Getting the type of 'sys_ssc' (line 352)
        sys_ssc_292096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'sys_ssc', False)
        # Obtaining the member 'to_discrete' of a type (line 352)
        to_discrete_292097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 18), sys_ssc_292096, 'to_discrete')
        # Calling to_discrete(args, kwargs) (line 352)
        to_discrete_call_result_292100 = invoke(stypy.reporting.localization.Localization(__file__, 352, 18), to_discrete_292097, *[float_292098], **kwargs_292099)
        
        # Assigning a type to the variable 'sys_ssd' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'sys_ssd', to_discrete_call_result_292100)
        
        # Call to assert_allclose(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'sys_ssd' (line 354)
        sys_ssd_292102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'sys_ssd', False)
        # Obtaining the member 'A' of a type (line 354)
        A_292103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 24), sys_ssd_292102, 'A')
        # Getting the type of 'A_res' (line 354)
        A_res_292104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'A_res', False)
        # Processing the call keyword arguments (line 354)
        kwargs_292105 = {}
        # Getting the type of 'assert_allclose' (line 354)
        assert_allclose_292101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 354)
        assert_allclose_call_result_292106 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), assert_allclose_292101, *[A_292103, A_res_292104], **kwargs_292105)
        
        
        # Call to assert_allclose(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'sys_ssd' (line 355)
        sys_ssd_292108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 24), 'sys_ssd', False)
        # Obtaining the member 'B' of a type (line 355)
        B_292109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 24), sys_ssd_292108, 'B')
        # Getting the type of 'B_res' (line 355)
        B_res_292110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'B_res', False)
        # Processing the call keyword arguments (line 355)
        kwargs_292111 = {}
        # Getting the type of 'assert_allclose' (line 355)
        assert_allclose_292107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 355)
        assert_allclose_call_result_292112 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), assert_allclose_292107, *[B_292109, B_res_292110], **kwargs_292111)
        
        
        # Call to assert_allclose(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'sys_ssd' (line 356)
        sys_ssd_292114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 24), 'sys_ssd', False)
        # Obtaining the member 'C' of a type (line 356)
        C_292115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 24), sys_ssd_292114, 'C')
        # Getting the type of 'C' (line 356)
        C_292116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 35), 'C', False)
        # Processing the call keyword arguments (line 356)
        kwargs_292117 = {}
        # Getting the type of 'assert_allclose' (line 356)
        assert_allclose_292113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 356)
        assert_allclose_call_result_292118 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), assert_allclose_292113, *[C_292115, C_292116], **kwargs_292117)
        
        
        # Call to assert_allclose(...): (line 357)
        # Processing the call arguments (line 357)
        # Getting the type of 'sys_ssd' (line 357)
        sys_ssd_292120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'sys_ssd', False)
        # Obtaining the member 'D' of a type (line 357)
        D_292121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 24), sys_ssd_292120, 'D')
        # Getting the type of 'D' (line 357)
        D_292122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'D', False)
        # Processing the call keyword arguments (line 357)
        kwargs_292123 = {}
        # Getting the type of 'assert_allclose' (line 357)
        assert_allclose_292119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 357)
        assert_allclose_call_result_292124 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), assert_allclose_292119, *[D_292121, D_292122], **kwargs_292123)
        
        
        # ################# End of 'test_c2d_ss(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_c2d_ss' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_292125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_c2d_ss'
        return stypy_return_type_292125


    @norecursion
    def test_c2d_tf(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_c2d_tf'
        module_type_store = module_type_store.open_function_context('test_c2d_tf', 359, 4, False)
        # Assigning a type to the variable 'self' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_localization', localization)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_function_name', 'TestC2dLti.test_c2d_tf')
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_param_names_list', [])
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestC2dLti.test_c2d_tf.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.test_c2d_tf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_c2d_tf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_c2d_tf(...)' code ##################

        
        # Assigning a Call to a Name (line 361):
        
        # Assigning a Call to a Name (line 361):
        
        # Call to lti(...): (line 361)
        # Processing the call arguments (line 361)
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_292127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        float_292128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 19), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 18), list_292127, float_292128)
        # Adding element type (line 361)
        float_292129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 18), list_292127, float_292129)
        
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_292130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        float_292131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 30), list_292130, float_292131)
        # Adding element type (line 361)
        float_292132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 36), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 30), list_292130, float_292132)
        
        # Processing the call keyword arguments (line 361)
        kwargs_292133 = {}
        # Getting the type of 'lti' (line 361)
        lti_292126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 14), 'lti', False)
        # Calling lti(args, kwargs) (line 361)
        lti_call_result_292134 = invoke(stypy.reporting.localization.Localization(__file__, 361, 14), lti_292126, *[list_292127, list_292130], **kwargs_292133)
        
        # Assigning a type to the variable 'sys' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'sys', lti_call_result_292134)
        
        # Assigning a Call to a Name (line 362):
        
        # Assigning a Call to a Name (line 362):
        
        # Call to to_discrete(...): (line 362)
        # Processing the call arguments (line 362)
        float_292137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 30), 'float')
        # Processing the call keyword arguments (line 362)
        kwargs_292138 = {}
        # Getting the type of 'sys' (line 362)
        sys_292135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 14), 'sys', False)
        # Obtaining the member 'to_discrete' of a type (line 362)
        to_discrete_292136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 14), sys_292135, 'to_discrete')
        # Calling to_discrete(args, kwargs) (line 362)
        to_discrete_call_result_292139 = invoke(stypy.reporting.localization.Localization(__file__, 362, 14), to_discrete_292136, *[float_292137], **kwargs_292138)
        
        # Assigning a type to the variable 'sys' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'sys', to_discrete_call_result_292139)
        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to array(...): (line 365)
        # Processing the call arguments (line 365)
        
        # Obtaining an instance of the builtin type 'list' (line 365)
        list_292142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 365)
        # Adding element type (line 365)
        float_292143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 27), list_292142, float_292143)
        # Adding element type (line 365)
        float_292144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 27), list_292142, float_292144)
        
        # Processing the call keyword arguments (line 365)
        kwargs_292145 = {}
        # Getting the type of 'np' (line 365)
        np_292140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 365)
        array_292141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 18), np_292140, 'array')
        # Calling array(args, kwargs) (line 365)
        array_call_result_292146 = invoke(stypy.reporting.localization.Localization(__file__, 365, 18), array_292141, *[list_292142], **kwargs_292145)
        
        # Assigning a type to the variable 'num_res' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'num_res', array_call_result_292146)
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to array(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Obtaining an instance of the builtin type 'list' (line 366)
        list_292149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 366)
        # Adding element type (line 366)
        float_292150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 27), list_292149, float_292150)
        # Adding element type (line 366)
        float_292151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 27), list_292149, float_292151)
        
        # Processing the call keyword arguments (line 366)
        kwargs_292152 = {}
        # Getting the type of 'np' (line 366)
        np_292147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 366)
        array_292148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 18), np_292147, 'array')
        # Calling array(args, kwargs) (line 366)
        array_call_result_292153 = invoke(stypy.reporting.localization.Localization(__file__, 366, 18), array_292148, *[list_292149], **kwargs_292152)
        
        # Assigning a type to the variable 'den_res' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'den_res', array_call_result_292153)
        
        # Call to assert_allclose(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'sys' (line 369)
        sys_292155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 24), 'sys', False)
        # Obtaining the member 'den' of a type (line 369)
        den_292156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 24), sys_292155, 'den')
        # Getting the type of 'den_res' (line 369)
        den_res_292157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 33), 'den_res', False)
        # Processing the call keyword arguments (line 369)
        float_292158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 47), 'float')
        keyword_292159 = float_292158
        kwargs_292160 = {'atol': keyword_292159}
        # Getting the type of 'assert_allclose' (line 369)
        assert_allclose_292154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 369)
        assert_allclose_call_result_292161 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), assert_allclose_292154, *[den_292156, den_res_292157], **kwargs_292160)
        
        
        # Call to assert_allclose(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'sys' (line 370)
        sys_292163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 24), 'sys', False)
        # Obtaining the member 'num' of a type (line 370)
        num_292164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 24), sys_292163, 'num')
        # Getting the type of 'num_res' (line 370)
        num_res_292165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 33), 'num_res', False)
        # Processing the call keyword arguments (line 370)
        float_292166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 47), 'float')
        keyword_292167 = float_292166
        kwargs_292168 = {'atol': keyword_292167}
        # Getting the type of 'assert_allclose' (line 370)
        assert_allclose_292162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 370)
        assert_allclose_call_result_292169 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert_allclose_292162, *[num_292164, num_res_292165], **kwargs_292168)
        
        
        # ################# End of 'test_c2d_tf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_c2d_tf' in the type store
        # Getting the type of 'stypy_return_type' (line 359)
        stypy_return_type_292170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_292170)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_c2d_tf'
        return stypy_return_type_292170


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 339, 0, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestC2dLti.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestC2dLti' (line 339)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 0), 'TestC2dLti', TestC2dLti)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
