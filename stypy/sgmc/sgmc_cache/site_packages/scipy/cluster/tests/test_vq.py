
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: from __future__ import division, print_function, absolute_import
3: 
4: import warnings
5: import sys
6: 
7: import numpy as np
8: from numpy.testing import (assert_array_equal, assert_array_almost_equal,
9:     assert_allclose, assert_equal, assert_)
10: from scipy._lib._numpy_compat import suppress_warnings
11: import pytest
12: from pytest import raises as assert_raises
13: 
14: from scipy.cluster.vq import (kmeans, kmeans2, py_vq, vq, whiten,
15:     ClusterError, _krandinit)
16: from scipy.cluster import _vq
17: 
18: 
19: TESTDATA_2D = np.array([
20:     -2.2, 1.17, -1.63, 1.69, -2.04, 4.38, -3.09, 0.95, -1.7, 4.79, -1.68, 0.68,
21:     -2.26, 3.34, -2.29, 2.55, -1.72, -0.72, -1.99, 2.34, -2.75, 3.43, -2.45,
22:     2.41, -4.26, 3.65, -1.57, 1.87, -1.96, 4.03, -3.01, 3.86, -2.53, 1.28,
23:     -4.0, 3.95, -1.62, 1.25, -3.42, 3.17, -1.17, 0.12, -3.03, -0.27, -2.07,
24:     -0.55, -1.17, 1.34, -2.82, 3.08, -2.44, 0.24, -1.71, 2.48, -5.23, 4.29,
25:     -2.08, 3.69, -1.89, 3.62, -2.09, 0.26, -0.92, 1.07, -2.25, 0.88, -2.25,
26:     2.02, -4.31, 3.86, -2.03, 3.42, -2.76, 0.3, -2.48, -0.29, -3.42, 3.21,
27:     -2.3, 1.73, -2.84, 0.69, -1.81, 2.48, -5.24, 4.52, -2.8, 1.31, -1.67,
28:     -2.34, -1.18, 2.17, -2.17, 2.82, -1.85, 2.25, -2.45, 1.86, -6.79, 3.94,
29:     -2.33, 1.89, -1.55, 2.08, -1.36, 0.93, -2.51, 2.74, -2.39, 3.92, -3.33,
30:     2.99, -2.06, -0.9, -2.83, 3.35, -2.59, 3.05, -2.36, 1.85, -1.69, 1.8,
31:     -1.39, 0.66, -2.06, 0.38, -1.47, 0.44, -4.68, 3.77, -5.58, 3.44, -2.29,
32:     2.24, -1.04, -0.38, -1.85, 4.23, -2.88, 0.73, -2.59, 1.39, -1.34, 1.75,
33:     -1.95, 1.3, -2.45, 3.09, -1.99, 3.41, -5.55, 5.21, -1.73, 2.52, -2.17,
34:     0.85, -2.06, 0.49, -2.54, 2.07, -2.03, 1.3, -3.23, 3.09, -1.55, 1.44,
35:     -0.81, 1.1, -2.99, 2.92, -1.59, 2.18, -2.45, -0.73, -3.12, -1.3, -2.83,
36:     0.2, -2.77, 3.24, -1.98, 1.6, -4.59, 3.39, -4.85, 3.75, -2.25, 1.71, -3.28,
37:     3.38, -1.74, 0.88, -2.41, 1.92, -2.24, 1.19, -2.48, 1.06, -1.68, -0.62,
38:     -1.3, 0.39, -1.78, 2.35, -3.54, 2.44, -1.32, 0.66, -2.38, 2.76, -2.35,
39:     3.95, -1.86, 4.32, -2.01, -1.23, -1.79, 2.76, -2.13, -0.13, -5.25, 3.84,
40:     -2.24, 1.59, -4.85, 2.96, -2.41, 0.01, -0.43, 0.13, -3.92, 2.91, -1.75,
41:     -0.53, -1.69, 1.69, -1.09, 0.15, -2.11, 2.17, -1.53, 1.22, -2.1, -0.86,
42:     -2.56, 2.28, -3.02, 3.33, -1.12, 3.86, -2.18, -1.19, -3.03, 0.79, -0.83,
43:     0.97, -3.19, 1.45, -1.34, 1.28, -2.52, 4.22, -4.53, 3.22, -1.97, 1.75,
44:     -2.36, 3.19, -0.83, 1.53, -1.59, 1.86, -2.17, 2.3, -1.63, 2.71, -2.03,
45:     3.75, -2.57, -0.6, -1.47, 1.33, -1.95, 0.7, -1.65, 1.27, -1.42, 1.09, -3.0,
46:     3.87, -2.51, 3.06, -2.6, 0.74, -1.08, -0.03, -2.44, 1.31, -2.65, 2.99,
47:     -1.84, 1.65, -4.76, 3.75, -2.07, 3.98, -2.4, 2.67, -2.21, 1.49, -1.21,
48:     1.22, -5.29, 2.38, -2.85, 2.28, -5.6, 3.78, -2.7, 0.8, -1.81, 3.5, -3.75,
49:     4.17, -1.29, 2.99, -5.92, 3.43, -1.83, 1.23, -1.24, -1.04, -2.56, 2.37,
50:     -3.26, 0.39, -4.63, 2.51, -4.52, 3.04, -1.7, 0.36, -1.41, 0.04, -2.1, 1.0,
51:     -1.87, 3.78, -4.32, 3.59, -2.24, 1.38, -1.99, -0.22, -1.87, 1.95, -0.84,
52:     2.17, -5.38, 3.56, -1.27, 2.9, -1.79, 3.31, -5.47, 3.85, -1.44, 3.69,
53:     -2.02, 0.37, -1.29, 0.33, -2.34, 2.56, -1.74, -1.27, -1.97, 1.22, -2.51,
54:     -0.16, -1.64, -0.96, -2.99, 1.4, -1.53, 3.31, -2.24, 0.45, -2.46, 1.71,
55:     -2.88, 1.56, -1.63, 1.46, -1.41, 0.68, -1.96, 2.76, -1.61,
56:     2.11]).reshape((200, 2))
57: 
58: 
59: # Global data
60: X = np.array([[3.0, 3], [4, 3], [4, 2],
61:                [9, 2], [5, 1], [6, 2], [9, 4],
62:                [5, 2], [5, 4], [7, 4], [6, 5]])
63: 
64: CODET1 = np.array([[3.0000, 3.0000],
65:                    [6.2000, 4.0000],
66:                    [5.8000, 1.8000]])
67: 
68: CODET2 = np.array([[11.0/3, 8.0/3],
69:                    [6.7500, 4.2500],
70:                    [6.2500, 1.7500]])
71: 
72: LABEL1 = np.array([0, 1, 2, 2, 2, 2, 1, 2, 1, 1, 1])
73: 
74: 
75: class TestWhiten(object):
76:     def test_whiten(self):
77:         desired = np.array([[5.08738849, 2.97091878],
78:                             [3.19909255, 0.69660580],
79:                             [4.51041982, 0.02640918],
80:                             [4.38567074, 0.95120889],
81:                             [2.32191480, 1.63195503]])
82:         for tp in np.array, np.matrix:
83:             obs = tp([[0.98744510, 0.82766775],
84:                       [0.62093317, 0.19406729],
85:                       [0.87545741, 0.00735733],
86:                       [0.85124403, 0.26499712],
87:                       [0.45067590, 0.45464607]])
88:             assert_allclose(whiten(obs), desired, rtol=1e-5)
89: 
90:     def test_whiten_zero_std(self):
91:         desired = np.array([[0., 1.0, 2.86666544],
92:                             [0., 1.0, 1.32460034],
93:                             [0., 1.0, 3.74382172]])
94:         for tp in np.array, np.matrix:
95:             obs = tp([[0., 1., 0.74109533],
96:                       [0., 1., 0.34243798],
97:                       [0., 1., 0.96785929]])
98:             with warnings.catch_warnings(record=True) as w:
99:                 warnings.simplefilter('always')
100:                 assert_allclose(whiten(obs), desired, rtol=1e-5)
101:                 assert_equal(len(w), 1)
102:                 assert_(issubclass(w[-1].category, RuntimeWarning))
103: 
104:     def test_whiten_not_finite(self):
105:         for tp in np.array, np.matrix:
106:             for bad_value in np.nan, np.inf, -np.inf:
107:                 obs = tp([[0.98744510, bad_value],
108:                           [0.62093317, 0.19406729],
109:                           [0.87545741, 0.00735733],
110:                           [0.85124403, 0.26499712],
111:                           [0.45067590, 0.45464607]])
112:                 assert_raises(ValueError, whiten, obs)
113: 
114: 
115: class TestVq(object):
116:     def test_py_vq(self):
117:         initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
118:         for tp in np.array, np.matrix:
119:             label1 = py_vq(tp(X), tp(initc))[0]
120:             assert_array_equal(label1, LABEL1)
121: 
122:     def test_vq(self):
123:         initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
124:         for tp in np.array, np.matrix:
125:             label1, dist = _vq.vq(tp(X), tp(initc))
126:             assert_array_equal(label1, LABEL1)
127:             tlabel1, tdist = vq(tp(X), tp(initc))
128: 
129:     def test_vq_1d(self):
130:         # Test special rank 1 vq algo, python implementation.
131:         data = X[:, 0]
132:         initc = data[:3]
133:         a, b = _vq.vq(data, initc)
134:         ta, tb = py_vq(data[:, np.newaxis], initc[:, np.newaxis])
135:         assert_array_equal(a, ta)
136:         assert_array_equal(b, tb)
137: 
138:     def test__vq_sametype(self):
139:         a = np.array([1.0, 2.0], dtype=np.float64)
140:         b = a.astype(np.float32)
141:         assert_raises(TypeError, _vq.vq, a, b)
142: 
143:     def test__vq_invalid_type(self):
144:         a = np.array([1, 2], dtype=int)
145:         assert_raises(TypeError, _vq.vq, a, a)
146: 
147:     def test_vq_large_nfeat(self):
148:         X = np.random.rand(20, 20)
149:         code_book = np.random.rand(3, 20)
150: 
151:         codes0, dis0 = _vq.vq(X, code_book)
152:         codes1, dis1 = py_vq(X, code_book)
153:         assert_allclose(dis0, dis1, 1e-5)
154:         assert_array_equal(codes0, codes1)
155: 
156:         X = X.astype(np.float32)
157:         code_book = code_book.astype(np.float32)
158: 
159:         codes0, dis0 = _vq.vq(X, code_book)
160:         codes1, dis1 = py_vq(X, code_book)
161:         assert_allclose(dis0, dis1, 1e-5)
162:         assert_array_equal(codes0, codes1)
163: 
164:     def test_vq_large_features(self):
165:         X = np.random.rand(10, 5) * 1000000
166:         code_book = np.random.rand(2, 5) * 1000000
167: 
168:         codes0, dis0 = _vq.vq(X, code_book)
169:         codes1, dis1 = py_vq(X, code_book)
170:         assert_allclose(dis0, dis1, 1e-5)
171:         assert_array_equal(codes0, codes1)
172: 
173: 
174: class TestKMean(object):
175:     def test_large_features(self):
176:         # Generate a data set with large values, and run kmeans on it to
177:         # (regression for 1077).
178:         d = 300
179:         n = 100
180: 
181:         m1 = np.random.randn(d)
182:         m2 = np.random.randn(d)
183:         x = 10000 * np.random.randn(n, d) - 20000 * m1
184:         y = 10000 * np.random.randn(n, d) + 20000 * m2
185: 
186:         data = np.empty((x.shape[0] + y.shape[0], d), np.double)
187:         data[:x.shape[0]] = x
188:         data[x.shape[0]:] = y
189: 
190:         kmeans(data, 2)
191: 
192:     def test_kmeans_simple(self):
193:         np.random.seed(54321)
194:         initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
195:         for tp in np.array, np.matrix:
196:             code1 = kmeans(tp(X), tp(initc), iter=1)[0]
197:             assert_array_almost_equal(code1, CODET2)
198: 
199:     def test_kmeans_lost_cluster(self):
200:         # This will cause kmeans to have a cluster with no points.
201:         data = TESTDATA_2D
202:         initk = np.array([[-1.8127404, -0.67128041],
203:                          [2.04621601, 0.07401111],
204:                          [-2.31149087,-0.05160469]])
205: 
206:         kmeans(data, initk)
207:         with suppress_warnings() as sup:
208:             sup.filter(UserWarning,
209:                        "One of the clusters is empty. Re-run kmeans with a "
210:                        "different initialization")
211:             kmeans2(data, initk, missing='warn')
212: 
213:         assert_raises(ClusterError, kmeans2, data, initk, missing='raise')
214: 
215:     def test_kmeans2_simple(self):
216:         np.random.seed(12345678)
217:         initc = np.concatenate(([[X[0]], [X[1]], [X[2]]]))
218:         for tp in np.array, np.matrix:
219:             code1 = kmeans2(tp(X), tp(initc), iter=1)[0]
220:             code2 = kmeans2(tp(X), tp(initc), iter=2)[0]
221: 
222:             assert_array_almost_equal(code1, CODET1)
223:             assert_array_almost_equal(code2, CODET2)
224: 
225:     def test_kmeans2_rank1(self):
226:         data = TESTDATA_2D
227:         data1 = data[:, 0]
228: 
229:         initc = data1[:3]
230:         code = initc.copy()
231:         kmeans2(data1, code, iter=1)[0]
232:         kmeans2(data1, code, iter=2)[0]
233: 
234:     def test_kmeans2_rank1_2(self):
235:         data = TESTDATA_2D
236:         data1 = data[:, 0]
237:         kmeans2(data1, 2, iter=1)
238: 
239:     def test_kmeans2_high_dim(self):
240:         # test kmeans2 when the number of dimensions exceeds the number
241:         # of input points
242:         data = TESTDATA_2D
243:         data = data.reshape((20, 20))[:10]
244:         kmeans2(data, 2)
245: 
246:     def test_kmeans2_init(self):
247:         np.random.seed(12345)
248:         data = TESTDATA_2D
249: 
250:         kmeans2(data, 3, minit='points')
251:         kmeans2(data[:, :1], 3, minit='points')  # special case (1-D)
252: 
253:         kmeans2(data, 3, minit='random')
254:         kmeans2(data[:, :1], 3, minit='random')  # special case (1-D)
255: 
256:     @pytest.mark.skipif(sys.platform == 'win32', reason='Fails with MemoryError in Wine.')
257:     def test_krandinit(self):
258:         data = TESTDATA_2D
259:         datas = [data.reshape((200, 2)), data.reshape((20, 20))[:10]]
260:         k = int(1e6)
261:         for data in datas:
262:             np.random.seed(1234)
263:             init = _krandinit(data, k)
264:             orig_cov = np.cov(data, rowvar=0)
265:             init_cov = np.cov(init, rowvar=0)
266:             assert_allclose(orig_cov, init_cov, atol=1e-2)
267: 
268:     def test_kmeans2_empty(self):
269:         # Regression test for gh-1032.
270:         assert_raises(ValueError, kmeans2, [], 2)
271: 
272:     def test_kmeans_0k(self):
273:         # Regression test for gh-1073: fail when k arg is 0.
274:         assert_raises(ValueError, kmeans, X, 0)
275:         assert_raises(ValueError, kmeans2, X, 0)
276:         assert_raises(ValueError, kmeans2, X, np.array([]))
277: 
278:     def test_kmeans_large_thres(self):
279:         # Regression test for gh-1774
280:         x = np.array([1,2,3,4,10], dtype=float)
281:         res = kmeans(x, 1, thresh=1e16)
282:         assert_allclose(res[0], np.array([4.]))
283:         assert_allclose(res[1], 2.3999999999999999)
284: 
285: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import warnings' statement (line 4)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11807 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_11807) is not StypyTypeError):

    if (import_11807 != 'pyd_module'):
        __import__(import_11807)
        sys_modules_11808 = sys.modules[import_11807]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_11808.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_11807)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose, assert_equal, assert_' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11809 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_11809) is not StypyTypeError):

    if (import_11809 != 'pyd_module'):
        __import__(import_11809)
        sys_modules_11810 = sys.modules[import_11809]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_11810.module_type_store, module_type_store, ['assert_array_equal', 'assert_array_almost_equal', 'assert_allclose', 'assert_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_11810, sys_modules_11810.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_allclose, assert_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_array_equal', 'assert_array_almost_equal', 'assert_allclose', 'assert_equal', 'assert_'], [assert_array_equal, assert_array_almost_equal, assert_allclose, assert_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_11809)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11811 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat')

if (type(import_11811) is not StypyTypeError):

    if (import_11811 != 'pyd_module'):
        __import__(import_11811)
        sys_modules_11812 = sys.modules[import_11811]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', sys_modules_11812.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_11812, sys_modules_11812.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._numpy_compat', import_11811)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import pytest' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11813 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest')

if (type(import_11813) is not StypyTypeError):

    if (import_11813 != 'pyd_module'):
        __import__(import_11813)
        sys_modules_11814 = sys.modules[import_11813]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', sys_modules_11814.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', import_11813)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from pytest import assert_raises' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11815 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest')

if (type(import_11815) is not StypyTypeError):

    if (import_11815 != 'pyd_module'):
        __import__(import_11815)
        sys_modules_11816 = sys.modules[import_11815]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', sys_modules_11816.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_11816, sys_modules_11816.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'pytest', import_11815)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.cluster.vq import kmeans, kmeans2, py_vq, vq, whiten, ClusterError, _krandinit' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.cluster.vq')

if (type(import_11817) is not StypyTypeError):

    if (import_11817 != 'pyd_module'):
        __import__(import_11817)
        sys_modules_11818 = sys.modules[import_11817]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.cluster.vq', sys_modules_11818.module_type_store, module_type_store, ['kmeans', 'kmeans2', 'py_vq', 'vq', 'whiten', 'ClusterError', '_krandinit'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_11818, sys_modules_11818.module_type_store, module_type_store)
    else:
        from scipy.cluster.vq import kmeans, kmeans2, py_vq, vq, whiten, ClusterError, _krandinit

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.cluster.vq', None, module_type_store, ['kmeans', 'kmeans2', 'py_vq', 'vq', 'whiten', 'ClusterError', '_krandinit'], [kmeans, kmeans2, py_vq, vq, whiten, ClusterError, _krandinit])

else:
    # Assigning a type to the variable 'scipy.cluster.vq' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.cluster.vq', import_11817)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.cluster import _vq' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/tests/')
import_11819 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.cluster')

if (type(import_11819) is not StypyTypeError):

    if (import_11819 != 'pyd_module'):
        __import__(import_11819)
        sys_modules_11820 = sys.modules[import_11819]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.cluster', sys_modules_11820.module_type_store, module_type_store, ['_vq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_11820, sys_modules_11820.module_type_store, module_type_store)
    else:
        from scipy.cluster import _vq

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.cluster', None, module_type_store, ['_vq'], [_vq])

else:
    # Assigning a type to the variable 'scipy.cluster' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.cluster', import_11819)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/tests/')


# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to reshape(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 56)
tuple_12227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 56)
# Adding element type (line 56)
int_12228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_12227, int_12228)
# Adding element type (line 56)
int_12229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), tuple_12227, int_12229)

# Processing the call keyword arguments (line 19)
kwargs_12230 = {}

# Call to array(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_11823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
float_11824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11824)
# Adding element type (line 19)
float_11825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11825)
# Adding element type (line 19)
float_11826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11826)
# Adding element type (line 19)
float_11827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11827)
# Adding element type (line 19)
float_11828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11828)
# Adding element type (line 19)
float_11829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11829)
# Adding element type (line 19)
float_11830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11830)
# Adding element type (line 19)
float_11831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11831)
# Adding element type (line 19)
float_11832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11832)
# Adding element type (line 19)
float_11833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 61), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11833)
# Adding element type (line 19)
float_11834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 67), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11834)
# Adding element type (line 19)
float_11835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 74), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11835)
# Adding element type (line 19)
float_11836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11836)
# Adding element type (line 19)
float_11837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11837)
# Adding element type (line 19)
float_11838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11838)
# Adding element type (line 19)
float_11839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11839)
# Adding element type (line 19)
float_11840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11840)
# Adding element type (line 19)
float_11841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11841)
# Adding element type (line 19)
float_11842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11842)
# Adding element type (line 19)
float_11843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 51), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11843)
# Adding element type (line 19)
float_11844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11844)
# Adding element type (line 19)
float_11845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11845)
# Adding element type (line 19)
float_11846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11846)
# Adding element type (line 19)
float_11847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11847)
# Adding element type (line 19)
float_11848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11848)
# Adding element type (line 19)
float_11849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11849)
# Adding element type (line 19)
float_11850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11850)
# Adding element type (line 19)
float_11851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11851)
# Adding element type (line 19)
float_11852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11852)
# Adding element type (line 19)
float_11853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11853)
# Adding element type (line 19)
float_11854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11854)
# Adding element type (line 19)
float_11855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11855)
# Adding element type (line 19)
float_11856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11856)
# Adding element type (line 19)
float_11857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11857)
# Adding element type (line 19)
float_11858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11858)
# Adding element type (line 19)
float_11859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11859)
# Adding element type (line 19)
float_11860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11860)
# Adding element type (line 19)
float_11861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11861)
# Adding element type (line 19)
float_11862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11862)
# Adding element type (line 19)
float_11863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11863)
# Adding element type (line 19)
float_11864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11864)
# Adding element type (line 19)
float_11865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11865)
# Adding element type (line 19)
float_11866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11866)
# Adding element type (line 19)
float_11867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11867)
# Adding element type (line 19)
float_11868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11868)
# Adding element type (line 19)
float_11869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11869)
# Adding element type (line 19)
float_11870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11870)
# Adding element type (line 19)
float_11871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11871)
# Adding element type (line 19)
float_11872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11872)
# Adding element type (line 19)
float_11873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11873)
# Adding element type (line 19)
float_11874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11874)
# Adding element type (line 19)
float_11875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11875)
# Adding element type (line 19)
float_11876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11876)
# Adding element type (line 19)
float_11877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11877)
# Adding element type (line 19)
float_11878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11878)
# Adding element type (line 19)
float_11879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11879)
# Adding element type (line 19)
float_11880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11880)
# Adding element type (line 19)
float_11881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11881)
# Adding element type (line 19)
float_11882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11882)
# Adding element type (line 19)
float_11883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11883)
# Adding element type (line 19)
float_11884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11884)
# Adding element type (line 19)
float_11885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11885)
# Adding element type (line 19)
float_11886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11886)
# Adding element type (line 19)
float_11887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11887)
# Adding element type (line 19)
float_11888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11888)
# Adding element type (line 19)
float_11889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11889)
# Adding element type (line 19)
float_11890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11890)
# Adding element type (line 19)
float_11891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11891)
# Adding element type (line 19)
float_11892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11892)
# Adding element type (line 19)
float_11893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11893)
# Adding element type (line 19)
float_11894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11894)
# Adding element type (line 19)
float_11895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11895)
# Adding element type (line 19)
float_11896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11896)
# Adding element type (line 19)
float_11897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11897)
# Adding element type (line 19)
float_11898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11898)
# Adding element type (line 19)
float_11899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11899)
# Adding element type (line 19)
float_11900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11900)
# Adding element type (line 19)
float_11901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11901)
# Adding element type (line 19)
float_11902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11902)
# Adding element type (line 19)
float_11903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11903)
# Adding element type (line 19)
float_11904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11904)
# Adding element type (line 19)
float_11905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11905)
# Adding element type (line 19)
float_11906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11906)
# Adding element type (line 19)
float_11907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11907)
# Adding element type (line 19)
float_11908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11908)
# Adding element type (line 19)
float_11909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11909)
# Adding element type (line 19)
float_11910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11910)
# Adding element type (line 19)
float_11911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 61), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11911)
# Adding element type (line 19)
float_11912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 67), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11912)
# Adding element type (line 19)
float_11913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11913)
# Adding element type (line 19)
float_11914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11914)
# Adding element type (line 19)
float_11915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11915)
# Adding element type (line 19)
float_11916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11916)
# Adding element type (line 19)
float_11917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11917)
# Adding element type (line 19)
float_11918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11918)
# Adding element type (line 19)
float_11919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11919)
# Adding element type (line 19)
float_11920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11920)
# Adding element type (line 19)
float_11921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11921)
# Adding element type (line 19)
float_11922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11922)
# Adding element type (line 19)
float_11923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11923)
# Adding element type (line 19)
float_11924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11924)
# Adding element type (line 19)
float_11925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11925)
# Adding element type (line 19)
float_11926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11926)
# Adding element type (line 19)
float_11927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11927)
# Adding element type (line 19)
float_11928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11928)
# Adding element type (line 19)
float_11929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11929)
# Adding element type (line 19)
float_11930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11930)
# Adding element type (line 19)
float_11931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11931)
# Adding element type (line 19)
float_11932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11932)
# Adding element type (line 19)
float_11933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11933)
# Adding element type (line 19)
float_11934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11934)
# Adding element type (line 19)
float_11935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11935)
# Adding element type (line 19)
float_11936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11936)
# Adding element type (line 19)
float_11937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11937)
# Adding element type (line 19)
float_11938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11938)
# Adding element type (line 19)
float_11939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11939)
# Adding element type (line 19)
float_11940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11940)
# Adding element type (line 19)
float_11941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11941)
# Adding element type (line 19)
float_11942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11942)
# Adding element type (line 19)
float_11943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11943)
# Adding element type (line 19)
float_11944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11944)
# Adding element type (line 19)
float_11945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11945)
# Adding element type (line 19)
float_11946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11946)
# Adding element type (line 19)
float_11947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11947)
# Adding element type (line 19)
float_11948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11948)
# Adding element type (line 19)
float_11949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11949)
# Adding element type (line 19)
float_11950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11950)
# Adding element type (line 19)
float_11951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11951)
# Adding element type (line 19)
float_11952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11952)
# Adding element type (line 19)
float_11953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11953)
# Adding element type (line 19)
float_11954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11954)
# Adding element type (line 19)
float_11955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11955)
# Adding element type (line 19)
float_11956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11956)
# Adding element type (line 19)
float_11957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11957)
# Adding element type (line 19)
float_11958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11958)
# Adding element type (line 19)
float_11959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11959)
# Adding element type (line 19)
float_11960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11960)
# Adding element type (line 19)
float_11961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11961)
# Adding element type (line 19)
float_11962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11962)
# Adding element type (line 19)
float_11963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11963)
# Adding element type (line 19)
float_11964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11964)
# Adding element type (line 19)
float_11965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11965)
# Adding element type (line 19)
float_11966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11966)
# Adding element type (line 19)
float_11967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11967)
# Adding element type (line 19)
float_11968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11968)
# Adding element type (line 19)
float_11969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11969)
# Adding element type (line 19)
float_11970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11970)
# Adding element type (line 19)
float_11971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11971)
# Adding element type (line 19)
float_11972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11972)
# Adding element type (line 19)
float_11973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11973)
# Adding element type (line 19)
float_11974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11974)
# Adding element type (line 19)
float_11975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11975)
# Adding element type (line 19)
float_11976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11976)
# Adding element type (line 19)
float_11977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11977)
# Adding element type (line 19)
float_11978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11978)
# Adding element type (line 19)
float_11979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11979)
# Adding element type (line 19)
float_11980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11980)
# Adding element type (line 19)
float_11981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11981)
# Adding element type (line 19)
float_11982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11982)
# Adding element type (line 19)
float_11983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11983)
# Adding element type (line 19)
float_11984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11984)
# Adding element type (line 19)
float_11985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11985)
# Adding element type (line 19)
float_11986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11986)
# Adding element type (line 19)
float_11987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11987)
# Adding element type (line 19)
float_11988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 61), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11988)
# Adding element type (line 19)
float_11989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11989)
# Adding element type (line 19)
float_11990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11990)
# Adding element type (line 19)
float_11991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11991)
# Adding element type (line 19)
float_11992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11992)
# Adding element type (line 19)
float_11993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11993)
# Adding element type (line 19)
float_11994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11994)
# Adding element type (line 19)
float_11995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11995)
# Adding element type (line 19)
float_11996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11996)
# Adding element type (line 19)
float_11997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11997)
# Adding element type (line 19)
float_11998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11998)
# Adding element type (line 19)
float_11999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_11999)
# Adding element type (line 19)
float_12000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12000)
# Adding element type (line 19)
float_12001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12001)
# Adding element type (line 19)
float_12002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12002)
# Adding element type (line 19)
float_12003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12003)
# Adding element type (line 19)
float_12004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12004)
# Adding element type (line 19)
float_12005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12005)
# Adding element type (line 19)
float_12006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12006)
# Adding element type (line 19)
float_12007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 41), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12007)
# Adding element type (line 19)
float_12008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12008)
# Adding element type (line 19)
float_12009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12009)
# Adding element type (line 19)
float_12010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 60), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12010)
# Adding element type (line 19)
float_12011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 67), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12011)
# Adding element type (line 19)
float_12012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 73), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12012)
# Adding element type (line 19)
float_12013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12013)
# Adding element type (line 19)
float_12014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12014)
# Adding element type (line 19)
float_12015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12015)
# Adding element type (line 19)
float_12016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12016)
# Adding element type (line 19)
float_12017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12017)
# Adding element type (line 19)
float_12018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12018)
# Adding element type (line 19)
float_12019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12019)
# Adding element type (line 19)
float_12020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12020)
# Adding element type (line 19)
float_12021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12021)
# Adding element type (line 19)
float_12022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12022)
# Adding element type (line 19)
float_12023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12023)
# Adding element type (line 19)
float_12024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12024)
# Adding element type (line 19)
float_12025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12025)
# Adding element type (line 19)
float_12026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12026)
# Adding element type (line 19)
float_12027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12027)
# Adding element type (line 19)
float_12028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12028)
# Adding element type (line 19)
float_12029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12029)
# Adding element type (line 19)
float_12030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12030)
# Adding element type (line 19)
float_12031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12031)
# Adding element type (line 19)
float_12032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12032)
# Adding element type (line 19)
float_12033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12033)
# Adding element type (line 19)
float_12034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12034)
# Adding element type (line 19)
float_12035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12035)
# Adding element type (line 19)
float_12036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12036)
# Adding element type (line 19)
float_12037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12037)
# Adding element type (line 19)
float_12038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12038)
# Adding element type (line 19)
float_12039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12039)
# Adding element type (line 19)
float_12040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12040)
# Adding element type (line 19)
float_12041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12041)
# Adding element type (line 19)
float_12042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12042)
# Adding element type (line 19)
float_12043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12043)
# Adding element type (line 19)
float_12044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12044)
# Adding element type (line 19)
float_12045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 71), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12045)
# Adding element type (line 19)
float_12046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12046)
# Adding element type (line 19)
float_12047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12047)
# Adding element type (line 19)
float_12048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12048)
# Adding element type (line 19)
float_12049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12049)
# Adding element type (line 19)
float_12050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12050)
# Adding element type (line 19)
float_12051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12051)
# Adding element type (line 19)
float_12052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12052)
# Adding element type (line 19)
float_12053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12053)
# Adding element type (line 19)
float_12054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12054)
# Adding element type (line 19)
float_12055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12055)
# Adding element type (line 19)
float_12056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12056)
# Adding element type (line 19)
float_12057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12057)
# Adding element type (line 19)
float_12058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12058)
# Adding element type (line 19)
float_12059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12059)
# Adding element type (line 19)
float_12060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12060)
# Adding element type (line 19)
float_12061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12061)
# Adding element type (line 19)
float_12062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12062)
# Adding element type (line 19)
float_12063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12063)
# Adding element type (line 19)
float_12064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12064)
# Adding element type (line 19)
float_12065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12065)
# Adding element type (line 19)
float_12066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12066)
# Adding element type (line 19)
float_12067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12067)
# Adding element type (line 19)
float_12068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12068)
# Adding element type (line 19)
float_12069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12069)
# Adding element type (line 19)
float_12070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12070)
# Adding element type (line 19)
float_12071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12071)
# Adding element type (line 19)
float_12072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12072)
# Adding element type (line 19)
float_12073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12073)
# Adding element type (line 19)
float_12074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12074)
# Adding element type (line 19)
float_12075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12075)
# Adding element type (line 19)
float_12076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12076)
# Adding element type (line 19)
float_12077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12077)
# Adding element type (line 19)
float_12078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12078)
# Adding element type (line 19)
float_12079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12079)
# Adding element type (line 19)
float_12080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12080)
# Adding element type (line 19)
float_12081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12081)
# Adding element type (line 19)
float_12082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12082)
# Adding element type (line 19)
float_12083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12083)
# Adding element type (line 19)
float_12084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12084)
# Adding element type (line 19)
float_12085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12085)
# Adding element type (line 19)
float_12086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12086)
# Adding element type (line 19)
float_12087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12087)
# Adding element type (line 19)
float_12088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12088)
# Adding element type (line 19)
float_12089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12089)
# Adding element type (line 19)
float_12090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12090)
# Adding element type (line 19)
float_12091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12091)
# Adding element type (line 19)
float_12092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12092)
# Adding element type (line 19)
float_12093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12093)
# Adding element type (line 19)
float_12094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12094)
# Adding element type (line 19)
float_12095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12095)
# Adding element type (line 19)
float_12096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12096)
# Adding element type (line 19)
float_12097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12097)
# Adding element type (line 19)
float_12098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12098)
# Adding element type (line 19)
float_12099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12099)
# Adding element type (line 19)
float_12100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12100)
# Adding element type (line 19)
float_12101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12101)
# Adding element type (line 19)
float_12102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12102)
# Adding element type (line 19)
float_12103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12103)
# Adding element type (line 19)
float_12104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12104)
# Adding element type (line 19)
float_12105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12105)
# Adding element type (line 19)
float_12106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12106)
# Adding element type (line 19)
float_12107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12107)
# Adding element type (line 19)
float_12108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12108)
# Adding element type (line 19)
float_12109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12109)
# Adding element type (line 19)
float_12110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 61), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12110)
# Adding element type (line 19)
float_12111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12111)
# Adding element type (line 19)
float_12112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 74), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12112)
# Adding element type (line 19)
float_12113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12113)
# Adding element type (line 19)
float_12114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12114)
# Adding element type (line 19)
float_12115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12115)
# Adding element type (line 19)
float_12116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12116)
# Adding element type (line 19)
float_12117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12117)
# Adding element type (line 19)
float_12118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12118)
# Adding element type (line 19)
float_12119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12119)
# Adding element type (line 19)
float_12120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12120)
# Adding element type (line 19)
float_12121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12121)
# Adding element type (line 19)
float_12122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12122)
# Adding element type (line 19)
float_12123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 69), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12123)
# Adding element type (line 19)
float_12124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12124)
# Adding element type (line 19)
float_12125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12125)
# Adding element type (line 19)
float_12126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12126)
# Adding element type (line 19)
float_12127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12127)
# Adding element type (line 19)
float_12128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12128)
# Adding element type (line 19)
float_12129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12129)
# Adding element type (line 19)
float_12130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12130)
# Adding element type (line 19)
float_12131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12131)
# Adding element type (line 19)
float_12132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12132)
# Adding element type (line 19)
float_12133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12133)
# Adding element type (line 19)
float_12134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12134)
# Adding element type (line 19)
float_12135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12135)
# Adding element type (line 19)
float_12136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12136)
# Adding element type (line 19)
float_12137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12137)
# Adding element type (line 19)
float_12138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12138)
# Adding element type (line 19)
float_12139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12139)
# Adding element type (line 19)
float_12140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12140)
# Adding element type (line 19)
float_12141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12141)
# Adding element type (line 19)
float_12142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12142)
# Adding element type (line 19)
float_12143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 54), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12143)
# Adding element type (line 19)
float_12144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 59), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12144)
# Adding element type (line 19)
float_12145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 66), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12145)
# Adding element type (line 19)
float_12146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 71), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12146)
# Adding element type (line 19)
float_12147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12147)
# Adding element type (line 19)
float_12148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12148)
# Adding element type (line 19)
float_12149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12149)
# Adding element type (line 19)
float_12150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12150)
# Adding element type (line 19)
float_12151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12151)
# Adding element type (line 19)
float_12152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12152)
# Adding element type (line 19)
float_12153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12153)
# Adding element type (line 19)
float_12154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12154)
# Adding element type (line 19)
float_12155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12155)
# Adding element type (line 19)
float_12156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12156)
# Adding element type (line 19)
float_12157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12157)
# Adding element type (line 19)
float_12158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12158)
# Adding element type (line 19)
float_12159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12159)
# Adding element type (line 19)
float_12160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12160)
# Adding element type (line 19)
float_12161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12161)
# Adding element type (line 19)
float_12162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12162)
# Adding element type (line 19)
float_12163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12163)
# Adding element type (line 19)
float_12164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12164)
# Adding element type (line 19)
float_12165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12165)
# Adding element type (line 19)
float_12166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12166)
# Adding element type (line 19)
float_12167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 62), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12167)
# Adding element type (line 19)
float_12168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12168)
# Adding element type (line 19)
float_12169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 74), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12169)
# Adding element type (line 19)
float_12170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12170)
# Adding element type (line 19)
float_12171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12171)
# Adding element type (line 19)
float_12172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12172)
# Adding element type (line 19)
float_12173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12173)
# Adding element type (line 19)
float_12174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12174)
# Adding element type (line 19)
float_12175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12175)
# Adding element type (line 19)
float_12176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12176)
# Adding element type (line 19)
float_12177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12177)
# Adding element type (line 19)
float_12178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12178)
# Adding element type (line 19)
float_12179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12179)
# Adding element type (line 19)
float_12180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12180)
# Adding element type (line 19)
float_12181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12181)
# Adding element type (line 19)
float_12182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12182)
# Adding element type (line 19)
float_12183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12183)
# Adding element type (line 19)
float_12184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12184)
# Adding element type (line 19)
float_12185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12185)
# Adding element type (line 19)
float_12186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12186)
# Adding element type (line 19)
float_12187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12187)
# Adding element type (line 19)
float_12188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12188)
# Adding element type (line 19)
float_12189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 55), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12189)
# Adding element type (line 19)
float_12190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 61), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12190)
# Adding element type (line 19)
float_12191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 68), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12191)
# Adding element type (line 19)
float_12192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12192)
# Adding element type (line 19)
float_12193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12193)
# Adding element type (line 19)
float_12194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12194)
# Adding element type (line 19)
float_12195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12195)
# Adding element type (line 19)
float_12196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12196)
# Adding element type (line 19)
float_12197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12197)
# Adding element type (line 19)
float_12198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12198)
# Adding element type (line 19)
float_12199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12199)
# Adding element type (line 19)
float_12200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12200)
# Adding element type (line 19)
float_12201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 64), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12201)
# Adding element type (line 19)
float_12202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12202)
# Adding element type (line 19)
float_12203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12203)
# Adding element type (line 19)
float_12204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12204)
# Adding element type (line 19)
float_12205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12205)
# Adding element type (line 19)
float_12206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12206)
# Adding element type (line 19)
float_12207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 32), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12207)
# Adding element type (line 19)
float_12208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12208)
# Adding element type (line 19)
float_12209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12209)
# Adding element type (line 19)
float_12210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12210)
# Adding element type (line 19)
float_12211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 57), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12211)
# Adding element type (line 19)
float_12212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 63), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12212)
# Adding element type (line 19)
float_12213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 70), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12213)
# Adding element type (line 19)
float_12214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12214)
# Adding element type (line 19)
float_12215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12215)
# Adding element type (line 19)
float_12216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12216)
# Adding element type (line 19)
float_12217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12217)
# Adding element type (line 19)
float_12218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12218)
# Adding element type (line 19)
float_12219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12219)
# Adding element type (line 19)
float_12220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12220)
# Adding element type (line 19)
float_12221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 50), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12221)
# Adding element type (line 19)
float_12222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 56), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12222)
# Adding element type (line 19)
float_12223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 23), list_11823, float_12223)

# Processing the call keyword arguments (line 19)
kwargs_12224 = {}
# Getting the type of 'np' (line 19)
np_11821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'np', False)
# Obtaining the member 'array' of a type (line 19)
array_11822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 14), np_11821, 'array')
# Calling array(args, kwargs) (line 19)
array_call_result_12225 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), array_11822, *[list_11823], **kwargs_12224)

# Obtaining the member 'reshape' of a type (line 19)
reshape_12226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 14), array_call_result_12225, 'reshape')
# Calling reshape(args, kwargs) (line 19)
reshape_call_result_12231 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), reshape_12226, *[tuple_12227], **kwargs_12230)

# Assigning a type to the variable 'TESTDATA_2D' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'TESTDATA_2D', reshape_call_result_12231)

# Assigning a Call to a Name (line 60):

# Assigning a Call to a Name (line 60):

# Call to array(...): (line 60)
# Processing the call arguments (line 60)

# Obtaining an instance of the builtin type 'list' (line 60)
list_12234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 60)
list_12235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
float_12236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 14), list_12235, float_12236)
# Adding element type (line 60)
int_12237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 14), list_12235, int_12237)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12235)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 60)
list_12238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
int_12239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 24), list_12238, int_12239)
# Adding element type (line 60)
int_12240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 24), list_12238, int_12240)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12238)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 60)
list_12241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
int_12242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 32), list_12241, int_12242)
# Adding element type (line 60)
int_12243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 32), list_12241, int_12243)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12241)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 61)
list_12244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
int_12245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 15), list_12244, int_12245)
# Adding element type (line 61)
int_12246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 15), list_12244, int_12246)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12244)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 61)
list_12247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
int_12248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), list_12247, int_12248)
# Adding element type (line 61)
int_12249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 23), list_12247, int_12249)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12247)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 61)
list_12250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
int_12251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 31), list_12250, int_12251)
# Adding element type (line 61)
int_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 31), list_12250, int_12252)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12250)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 61)
list_12253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
int_12254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), list_12253, int_12254)
# Adding element type (line 61)
int_12255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 39), list_12253, int_12255)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12253)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 62)
list_12256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
int_12257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_12256, int_12257)
# Adding element type (line 62)
int_12258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 15), list_12256, int_12258)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12256)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 62)
list_12259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
int_12260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 23), list_12259, int_12260)
# Adding element type (line 62)
int_12261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 23), list_12259, int_12261)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12259)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 62)
list_12262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 31), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
int_12263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 31), list_12262, int_12263)
# Adding element type (line 62)
int_12264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 31), list_12262, int_12264)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12262)
# Adding element type (line 60)

# Obtaining an instance of the builtin type 'list' (line 62)
list_12265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
int_12266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 39), list_12265, int_12266)
# Adding element type (line 62)
int_12267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 39), list_12265, int_12267)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 13), list_12234, list_12265)

# Processing the call keyword arguments (line 60)
kwargs_12268 = {}
# Getting the type of 'np' (line 60)
np_12232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'np', False)
# Obtaining the member 'array' of a type (line 60)
array_12233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 4), np_12232, 'array')
# Calling array(args, kwargs) (line 60)
array_call_result_12269 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), array_12233, *[list_12234], **kwargs_12268)

# Assigning a type to the variable 'X' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'X', array_call_result_12269)

# Assigning a Call to a Name (line 64):

# Assigning a Call to a Name (line 64):

# Call to array(...): (line 64)
# Processing the call arguments (line 64)

# Obtaining an instance of the builtin type 'list' (line 64)
list_12272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 64)
# Adding element type (line 64)

# Obtaining an instance of the builtin type 'list' (line 64)
list_12273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 64)
# Adding element type (line 64)
float_12274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_12273, float_12274)
# Adding element type (line 64)
float_12275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_12273, float_12275)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_12272, list_12273)
# Adding element type (line 64)

# Obtaining an instance of the builtin type 'list' (line 65)
list_12276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 65)
# Adding element type (line 65)
float_12277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_12276, float_12277)
# Adding element type (line 65)
float_12278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 19), list_12276, float_12278)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_12272, list_12276)
# Adding element type (line 64)

# Obtaining an instance of the builtin type 'list' (line 66)
list_12279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 66)
# Adding element type (line 66)
float_12280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), list_12279, float_12280)
# Adding element type (line 66)
float_12281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), list_12279, float_12281)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 18), list_12272, list_12279)

# Processing the call keyword arguments (line 64)
kwargs_12282 = {}
# Getting the type of 'np' (line 64)
np_12270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'np', False)
# Obtaining the member 'array' of a type (line 64)
array_12271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 9), np_12270, 'array')
# Calling array(args, kwargs) (line 64)
array_call_result_12283 = invoke(stypy.reporting.localization.Localization(__file__, 64, 9), array_12271, *[list_12272], **kwargs_12282)

# Assigning a type to the variable 'CODET1' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'CODET1', array_call_result_12283)

# Assigning a Call to a Name (line 68):

# Assigning a Call to a Name (line 68):

# Call to array(...): (line 68)
# Processing the call arguments (line 68)

# Obtaining an instance of the builtin type 'list' (line 68)
list_12286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 68)
# Adding element type (line 68)

# Obtaining an instance of the builtin type 'list' (line 68)
list_12287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 68)
# Adding element type (line 68)
float_12288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'float')
int_12289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
# Applying the binary operator 'div' (line 68)
result_div_12290 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 20), 'div', float_12288, int_12289)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_12287, result_div_12290)
# Adding element type (line 68)
float_12291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'float')
int_12292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 32), 'int')
# Applying the binary operator 'div' (line 68)
result_div_12293 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 28), 'div', float_12291, int_12292)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_12287, result_div_12293)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_12286, list_12287)
# Adding element type (line 68)

# Obtaining an instance of the builtin type 'list' (line 69)
list_12294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
float_12295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_12294, float_12295)
# Adding element type (line 69)
float_12296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), list_12294, float_12296)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_12286, list_12294)
# Adding element type (line 68)

# Obtaining an instance of the builtin type 'list' (line 70)
list_12297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 70)
# Adding element type (line 70)
float_12298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 19), list_12297, float_12298)
# Adding element type (line 70)
float_12299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 19), list_12297, float_12299)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 18), list_12286, list_12297)

# Processing the call keyword arguments (line 68)
kwargs_12300 = {}
# Getting the type of 'np' (line 68)
np_12284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 9), 'np', False)
# Obtaining the member 'array' of a type (line 68)
array_12285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 9), np_12284, 'array')
# Calling array(args, kwargs) (line 68)
array_call_result_12301 = invoke(stypy.reporting.localization.Localization(__file__, 68, 9), array_12285, *[list_12286], **kwargs_12300)

# Assigning a type to the variable 'CODET2' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'CODET2', array_call_result_12301)

# Assigning a Call to a Name (line 72):

# Assigning a Call to a Name (line 72):

# Call to array(...): (line 72)
# Processing the call arguments (line 72)

# Obtaining an instance of the builtin type 'list' (line 72)
list_12304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 72)
# Adding element type (line 72)
int_12305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12305)
# Adding element type (line 72)
int_12306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12306)
# Adding element type (line 72)
int_12307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12307)
# Adding element type (line 72)
int_12308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12308)
# Adding element type (line 72)
int_12309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12309)
# Adding element type (line 72)
int_12310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12310)
# Adding element type (line 72)
int_12311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12311)
# Adding element type (line 72)
int_12312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12312)
# Adding element type (line 72)
int_12313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12313)
# Adding element type (line 72)
int_12314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12314)
# Adding element type (line 72)
int_12315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 49), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 18), list_12304, int_12315)

# Processing the call keyword arguments (line 72)
kwargs_12316 = {}
# Getting the type of 'np' (line 72)
np_12302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'np', False)
# Obtaining the member 'array' of a type (line 72)
array_12303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 9), np_12302, 'array')
# Calling array(args, kwargs) (line 72)
array_call_result_12317 = invoke(stypy.reporting.localization.Localization(__file__, 72, 9), array_12303, *[list_12304], **kwargs_12316)

# Assigning a type to the variable 'LABEL1' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'LABEL1', array_call_result_12317)
# Declaration of the 'TestWhiten' class

class TestWhiten(object, ):

    @norecursion
    def test_whiten(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_whiten'
        module_type_store = module_type_store.open_function_context('test_whiten', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_localization', localization)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_function_name', 'TestWhiten.test_whiten')
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_param_names_list', [])
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWhiten.test_whiten.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWhiten.test_whiten', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_whiten', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_whiten(...)' code ##################

        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to array(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_12320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_12321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        # Adding element type (line 77)
        float_12322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 28), list_12321, float_12322)
        # Adding element type (line 77)
        float_12323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 28), list_12321, float_12323)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), list_12320, list_12321)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_12324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        # Adding element type (line 78)
        float_12325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 28), list_12324, float_12325)
        # Adding element type (line 78)
        float_12326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 28), list_12324, float_12326)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), list_12320, list_12324)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_12327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        float_12328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 28), list_12327, float_12328)
        # Adding element type (line 79)
        float_12329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 28), list_12327, float_12329)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), list_12320, list_12327)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_12330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        float_12331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), list_12330, float_12331)
        # Adding element type (line 80)
        float_12332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), list_12330, float_12332)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), list_12320, list_12330)
        # Adding element type (line 77)
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_12333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        # Adding element type (line 81)
        float_12334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), list_12333, float_12334)
        # Adding element type (line 81)
        float_12335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 41), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 28), list_12333, float_12335)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 27), list_12320, list_12333)
        
        # Processing the call keyword arguments (line 77)
        kwargs_12336 = {}
        # Getting the type of 'np' (line 77)
        np_12318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 77)
        array_12319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 18), np_12318, 'array')
        # Calling array(args, kwargs) (line 77)
        array_call_result_12337 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), array_12319, *[list_12320], **kwargs_12336)
        
        # Assigning a type to the variable 'desired' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'desired', array_call_result_12337)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_12338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'np' (line 82)
        np_12339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 18), 'np')
        # Obtaining the member 'array' of a type (line 82)
        array_12340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 18), np_12339, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_12338, array_12340)
        # Adding element type (line 82)
        # Getting the type of 'np' (line 82)
        np_12341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 82)
        matrix_12342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 28), np_12341, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 18), tuple_12338, matrix_12342)
        
        # Testing the type of a for loop iterable (line 82)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 8), tuple_12338)
        # Getting the type of the for loop variable (line 82)
        for_loop_var_12343 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 8), tuple_12338)
        # Assigning a type to the variable 'tp' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tp', for_loop_var_12343)
        # SSA begins for a for statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to tp(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_12345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_12346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        float_12347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_12346, float_12347)
        # Adding element type (line 83)
        float_12348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 22), list_12346, float_12348)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_12345, list_12346)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 84)
        list_12349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 84)
        # Adding element type (line 84)
        float_12350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_12349, float_12350)
        # Adding element type (line 84)
        float_12351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 22), list_12349, float_12351)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_12345, list_12349)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_12352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        float_12353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_12352, float_12353)
        # Adding element type (line 85)
        float_12354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 22), list_12352, float_12354)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_12345, list_12352)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 86)
        list_12355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 86)
        # Adding element type (line 86)
        float_12356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), list_12355, float_12356)
        # Adding element type (line 86)
        float_12357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 22), list_12355, float_12357)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_12345, list_12355)
        # Adding element type (line 83)
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_12358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        float_12359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 22), list_12358, float_12359)
        # Adding element type (line 87)
        float_12360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 22), list_12358, float_12360)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 21), list_12345, list_12358)
        
        # Processing the call keyword arguments (line 83)
        kwargs_12361 = {}
        # Getting the type of 'tp' (line 83)
        tp_12344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'tp', False)
        # Calling tp(args, kwargs) (line 83)
        tp_call_result_12362 = invoke(stypy.reporting.localization.Localization(__file__, 83, 18), tp_12344, *[list_12345], **kwargs_12361)
        
        # Assigning a type to the variable 'obs' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'obs', tp_call_result_12362)
        
        # Call to assert_allclose(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to whiten(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'obs' (line 88)
        obs_12365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'obs', False)
        # Processing the call keyword arguments (line 88)
        kwargs_12366 = {}
        # Getting the type of 'whiten' (line 88)
        whiten_12364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'whiten', False)
        # Calling whiten(args, kwargs) (line 88)
        whiten_call_result_12367 = invoke(stypy.reporting.localization.Localization(__file__, 88, 28), whiten_12364, *[obs_12365], **kwargs_12366)
        
        # Getting the type of 'desired' (line 88)
        desired_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 41), 'desired', False)
        # Processing the call keyword arguments (line 88)
        float_12369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 55), 'float')
        keyword_12370 = float_12369
        kwargs_12371 = {'rtol': keyword_12370}
        # Getting the type of 'assert_allclose' (line 88)
        assert_allclose_12363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 88)
        assert_allclose_call_result_12372 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), assert_allclose_12363, *[whiten_call_result_12367, desired_12368], **kwargs_12371)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_whiten(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_whiten' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12373)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_whiten'
        return stypy_return_type_12373


    @norecursion
    def test_whiten_zero_std(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_whiten_zero_std'
        module_type_store = module_type_store.open_function_context('test_whiten_zero_std', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_localization', localization)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_function_name', 'TestWhiten.test_whiten_zero_std')
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_param_names_list', [])
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWhiten.test_whiten_zero_std.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWhiten.test_whiten_zero_std', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_whiten_zero_std', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_whiten_zero_std(...)' code ##################

        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to array(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_12376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_12377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        # Adding element type (line 91)
        float_12378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), list_12377, float_12378)
        # Adding element type (line 91)
        float_12379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), list_12377, float_12379)
        # Adding element type (line 91)
        float_12380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 28), list_12377, float_12380)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 27), list_12376, list_12377)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_12381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        float_12382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_12381, float_12382)
        # Adding element type (line 92)
        float_12383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_12381, float_12383)
        # Adding element type (line 92)
        float_12384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 28), list_12381, float_12384)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 27), list_12376, list_12381)
        # Adding element type (line 91)
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_12385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        float_12386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_12385, float_12386)
        # Adding element type (line 93)
        float_12387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_12385, float_12387)
        # Adding element type (line 93)
        float_12388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 28), list_12385, float_12388)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 27), list_12376, list_12385)
        
        # Processing the call keyword arguments (line 91)
        kwargs_12389 = {}
        # Getting the type of 'np' (line 91)
        np_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'np', False)
        # Obtaining the member 'array' of a type (line 91)
        array_12375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 18), np_12374, 'array')
        # Calling array(args, kwargs) (line 91)
        array_call_result_12390 = invoke(stypy.reporting.localization.Localization(__file__, 91, 18), array_12375, *[list_12376], **kwargs_12389)
        
        # Assigning a type to the variable 'desired' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'desired', array_call_result_12390)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 94)
        tuple_12391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 94)
        # Adding element type (line 94)
        # Getting the type of 'np' (line 94)
        np_12392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'np')
        # Obtaining the member 'array' of a type (line 94)
        array_12393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 18), np_12392, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 18), tuple_12391, array_12393)
        # Adding element type (line 94)
        # Getting the type of 'np' (line 94)
        np_12394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 94)
        matrix_12395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), np_12394, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 18), tuple_12391, matrix_12395)
        
        # Testing the type of a for loop iterable (line 94)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 8), tuple_12391)
        # Getting the type of the for loop variable (line 94)
        for_loop_var_12396 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 8), tuple_12391)
        # Assigning a type to the variable 'tp' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tp', for_loop_var_12396)
        # SSA begins for a for statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to tp(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_12398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 95)
        list_12399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 95)
        # Adding element type (line 95)
        float_12400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_12399, float_12400)
        # Adding element type (line 95)
        float_12401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_12399, float_12401)
        # Adding element type (line 95)
        float_12402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 22), list_12399, float_12402)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_12398, list_12399)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_12403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        float_12404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 22), list_12403, float_12404)
        # Adding element type (line 96)
        float_12405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 22), list_12403, float_12405)
        # Adding element type (line 96)
        float_12406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 22), list_12403, float_12406)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_12398, list_12403)
        # Adding element type (line 95)
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_12407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        # Adding element type (line 97)
        float_12408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 22), list_12407, float_12408)
        # Adding element type (line 97)
        float_12409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 22), list_12407, float_12409)
        # Adding element type (line 97)
        float_12410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 22), list_12407, float_12410)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_12398, list_12407)
        
        # Processing the call keyword arguments (line 95)
        kwargs_12411 = {}
        # Getting the type of 'tp' (line 95)
        tp_12397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'tp', False)
        # Calling tp(args, kwargs) (line 95)
        tp_call_result_12412 = invoke(stypy.reporting.localization.Localization(__file__, 95, 18), tp_12397, *[list_12398], **kwargs_12411)
        
        # Assigning a type to the variable 'obs' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'obs', tp_call_result_12412)
        
        # Call to catch_warnings(...): (line 98)
        # Processing the call keyword arguments (line 98)
        # Getting the type of 'True' (line 98)
        True_12415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 48), 'True', False)
        keyword_12416 = True_12415
        kwargs_12417 = {'record': keyword_12416}
        # Getting the type of 'warnings' (line 98)
        warnings_12413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'warnings', False)
        # Obtaining the member 'catch_warnings' of a type (line 98)
        catch_warnings_12414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), warnings_12413, 'catch_warnings')
        # Calling catch_warnings(args, kwargs) (line 98)
        catch_warnings_call_result_12418 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), catch_warnings_12414, *[], **kwargs_12417)
        
        with_12419 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 98, 17), catch_warnings_call_result_12418, 'with parameter', '__enter__', '__exit__')

        if with_12419:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 98)
            enter___12420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), catch_warnings_call_result_12418, '__enter__')
            with_enter_12421 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), enter___12420)
            # Assigning a type to the variable 'w' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'w', with_enter_12421)
            
            # Call to simplefilter(...): (line 99)
            # Processing the call arguments (line 99)
            str_12424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'str', 'always')
            # Processing the call keyword arguments (line 99)
            kwargs_12425 = {}
            # Getting the type of 'warnings' (line 99)
            warnings_12422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 99)
            simplefilter_12423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), warnings_12422, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 99)
            simplefilter_call_result_12426 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), simplefilter_12423, *[str_12424], **kwargs_12425)
            
            
            # Call to assert_allclose(...): (line 100)
            # Processing the call arguments (line 100)
            
            # Call to whiten(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'obs' (line 100)
            obs_12429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'obs', False)
            # Processing the call keyword arguments (line 100)
            kwargs_12430 = {}
            # Getting the type of 'whiten' (line 100)
            whiten_12428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'whiten', False)
            # Calling whiten(args, kwargs) (line 100)
            whiten_call_result_12431 = invoke(stypy.reporting.localization.Localization(__file__, 100, 32), whiten_12428, *[obs_12429], **kwargs_12430)
            
            # Getting the type of 'desired' (line 100)
            desired_12432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'desired', False)
            # Processing the call keyword arguments (line 100)
            float_12433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 59), 'float')
            keyword_12434 = float_12433
            kwargs_12435 = {'rtol': keyword_12434}
            # Getting the type of 'assert_allclose' (line 100)
            assert_allclose_12427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'assert_allclose', False)
            # Calling assert_allclose(args, kwargs) (line 100)
            assert_allclose_call_result_12436 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), assert_allclose_12427, *[whiten_call_result_12431, desired_12432], **kwargs_12435)
            
            
            # Call to assert_equal(...): (line 101)
            # Processing the call arguments (line 101)
            
            # Call to len(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'w' (line 101)
            w_12439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'w', False)
            # Processing the call keyword arguments (line 101)
            kwargs_12440 = {}
            # Getting the type of 'len' (line 101)
            len_12438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 29), 'len', False)
            # Calling len(args, kwargs) (line 101)
            len_call_result_12441 = invoke(stypy.reporting.localization.Localization(__file__, 101, 29), len_12438, *[w_12439], **kwargs_12440)
            
            int_12442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 37), 'int')
            # Processing the call keyword arguments (line 101)
            kwargs_12443 = {}
            # Getting the type of 'assert_equal' (line 101)
            assert_equal_12437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'assert_equal', False)
            # Calling assert_equal(args, kwargs) (line 101)
            assert_equal_call_result_12444 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), assert_equal_12437, *[len_call_result_12441, int_12442], **kwargs_12443)
            
            
            # Call to assert_(...): (line 102)
            # Processing the call arguments (line 102)
            
            # Call to issubclass(...): (line 102)
            # Processing the call arguments (line 102)
            
            # Obtaining the type of the subscript
            int_12447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'int')
            # Getting the type of 'w' (line 102)
            w_12448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'w', False)
            # Obtaining the member '__getitem__' of a type (line 102)
            getitem___12449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), w_12448, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 102)
            subscript_call_result_12450 = invoke(stypy.reporting.localization.Localization(__file__, 102, 35), getitem___12449, int_12447)
            
            # Obtaining the member 'category' of a type (line 102)
            category_12451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 35), subscript_call_result_12450, 'category')
            # Getting the type of 'RuntimeWarning' (line 102)
            RuntimeWarning_12452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'RuntimeWarning', False)
            # Processing the call keyword arguments (line 102)
            kwargs_12453 = {}
            # Getting the type of 'issubclass' (line 102)
            issubclass_12446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'issubclass', False)
            # Calling issubclass(args, kwargs) (line 102)
            issubclass_call_result_12454 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), issubclass_12446, *[category_12451, RuntimeWarning_12452], **kwargs_12453)
            
            # Processing the call keyword arguments (line 102)
            kwargs_12455 = {}
            # Getting the type of 'assert_' (line 102)
            assert__12445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 102)
            assert__call_result_12456 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), assert__12445, *[issubclass_call_result_12454], **kwargs_12455)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 98)
            exit___12457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), catch_warnings_call_result_12418, '__exit__')
            with_exit_12458 = invoke(stypy.reporting.localization.Localization(__file__, 98, 17), exit___12457, None, None, None)

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_whiten_zero_std(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_whiten_zero_std' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_12459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12459)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_whiten_zero_std'
        return stypy_return_type_12459


    @norecursion
    def test_whiten_not_finite(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_whiten_not_finite'
        module_type_store = module_type_store.open_function_context('test_whiten_not_finite', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_localization', localization)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_function_name', 'TestWhiten.test_whiten_not_finite')
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_param_names_list', [])
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestWhiten.test_whiten_not_finite.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWhiten.test_whiten_not_finite', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_whiten_not_finite', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_whiten_not_finite(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_12460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        # Getting the type of 'np' (line 105)
        np_12461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'np')
        # Obtaining the member 'array' of a type (line 105)
        array_12462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 18), np_12461, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 18), tuple_12460, array_12462)
        # Adding element type (line 105)
        # Getting the type of 'np' (line 105)
        np_12463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 105)
        matrix_12464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), np_12463, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 18), tuple_12460, matrix_12464)
        
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), tuple_12460)
        # Getting the type of the for loop variable (line 105)
        for_loop_var_12465 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), tuple_12460)
        # Assigning a type to the variable 'tp' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tp', for_loop_var_12465)
        # SSA begins for a for statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_12466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'np' (line 106)
        np_12467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'np')
        # Obtaining the member 'nan' of a type (line 106)
        nan_12468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 29), np_12467, 'nan')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 29), tuple_12466, nan_12468)
        # Adding element type (line 106)
        # Getting the type of 'np' (line 106)
        np_12469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'np')
        # Obtaining the member 'inf' of a type (line 106)
        inf_12470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 37), np_12469, 'inf')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 29), tuple_12466, inf_12470)
        # Adding element type (line 106)
        
        # Getting the type of 'np' (line 106)
        np_12471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 46), 'np')
        # Obtaining the member 'inf' of a type (line 106)
        inf_12472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 46), np_12471, 'inf')
        # Applying the 'usub' unary operator (line 106)
        result___neg___12473 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 45), 'usub', inf_12472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 29), tuple_12466, result___neg___12473)
        
        # Testing the type of a for loop iterable (line 106)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 12), tuple_12466)
        # Getting the type of the for loop variable (line 106)
        for_loop_var_12474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 12), tuple_12466)
        # Assigning a type to the variable 'bad_value' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'bad_value', for_loop_var_12474)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 107):
        
        # Assigning a Call to a Name (line 107):
        
        # Call to tp(...): (line 107)
        # Processing the call arguments (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_12476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_12477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        # Adding element type (line 107)
        float_12478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), list_12477, float_12478)
        # Adding element type (line 107)
        # Getting the type of 'bad_value' (line 107)
        bad_value_12479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 39), 'bad_value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 26), list_12477, bad_value_12479)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_12476, list_12477)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_12480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        float_12481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_12480, float_12481)
        # Adding element type (line 108)
        float_12482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 26), list_12480, float_12482)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_12476, list_12480)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_12483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        # Adding element type (line 109)
        float_12484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 26), list_12483, float_12484)
        # Adding element type (line 109)
        float_12485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 26), list_12483, float_12485)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_12476, list_12483)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 110)
        list_12486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 110)
        # Adding element type (line 110)
        float_12487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 26), list_12486, float_12487)
        # Adding element type (line 110)
        float_12488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 26), list_12486, float_12488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_12476, list_12486)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'list' (line 111)
        list_12489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 111)
        # Adding element type (line 111)
        float_12490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 26), list_12489, float_12490)
        # Adding element type (line 111)
        float_12491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 26), list_12489, float_12491)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 25), list_12476, list_12489)
        
        # Processing the call keyword arguments (line 107)
        kwargs_12492 = {}
        # Getting the type of 'tp' (line 107)
        tp_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'tp', False)
        # Calling tp(args, kwargs) (line 107)
        tp_call_result_12493 = invoke(stypy.reporting.localization.Localization(__file__, 107, 22), tp_12475, *[list_12476], **kwargs_12492)
        
        # Assigning a type to the variable 'obs' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'obs', tp_call_result_12493)
        
        # Call to assert_raises(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'ValueError' (line 112)
        ValueError_12495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'ValueError', False)
        # Getting the type of 'whiten' (line 112)
        whiten_12496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 42), 'whiten', False)
        # Getting the type of 'obs' (line 112)
        obs_12497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 50), 'obs', False)
        # Processing the call keyword arguments (line 112)
        kwargs_12498 = {}
        # Getting the type of 'assert_raises' (line 112)
        assert_raises_12494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 112)
        assert_raises_call_result_12499 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), assert_raises_12494, *[ValueError_12495, whiten_12496, obs_12497], **kwargs_12498)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_whiten_not_finite(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_whiten_not_finite' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_12500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12500)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_whiten_not_finite'
        return stypy_return_type_12500


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 75, 0, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestWhiten.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestWhiten' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'TestWhiten', TestWhiten)
# Declaration of the 'TestVq' class

class TestVq(object, ):

    @norecursion
    def test_py_vq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_py_vq'
        module_type_store = module_type_store.open_function_context('test_py_vq', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test_py_vq.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_function_name', 'TestVq.test_py_vq')
        TestVq.test_py_vq.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test_py_vq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test_py_vq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test_py_vq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_py_vq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_py_vq(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to concatenate(...): (line 117)
        # Processing the call arguments (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_12503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_12504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining the type of the subscript
        int_12505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 36), 'int')
        # Getting the type of 'X' (line 117)
        X_12506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 34), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___12507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 34), X_12506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_12508 = invoke(stypy.reporting.localization.Localization(__file__, 117, 34), getitem___12507, int_12505)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 33), list_12504, subscript_call_result_12508)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 32), list_12503, list_12504)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_12509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining the type of the subscript
        int_12510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'int')
        # Getting the type of 'X' (line 117)
        X_12511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 42), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___12512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 42), X_12511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_12513 = invoke(stypy.reporting.localization.Localization(__file__, 117, 42), getitem___12512, int_12510)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 41), list_12509, subscript_call_result_12513)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 32), list_12503, list_12509)
        # Adding element type (line 117)
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_12514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        # Adding element type (line 117)
        
        # Obtaining the type of the subscript
        int_12515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 52), 'int')
        # Getting the type of 'X' (line 117)
        X_12516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 50), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___12517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 50), X_12516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_12518 = invoke(stypy.reporting.localization.Localization(__file__, 117, 50), getitem___12517, int_12515)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 49), list_12514, subscript_call_result_12518)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 32), list_12503, list_12514)
        
        # Processing the call keyword arguments (line 117)
        kwargs_12519 = {}
        # Getting the type of 'np' (line 117)
        np_12501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 117)
        concatenate_12502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), np_12501, 'concatenate')
        # Calling concatenate(args, kwargs) (line 117)
        concatenate_call_result_12520 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), concatenate_12502, *[list_12503], **kwargs_12519)
        
        # Assigning a type to the variable 'initc' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'initc', concatenate_call_result_12520)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_12521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        # Getting the type of 'np' (line 118)
        np_12522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'np')
        # Obtaining the member 'array' of a type (line 118)
        array_12523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 18), np_12522, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_12521, array_12523)
        # Adding element type (line 118)
        # Getting the type of 'np' (line 118)
        np_12524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 118)
        matrix_12525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), np_12524, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 18), tuple_12521, matrix_12525)
        
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), tuple_12521)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_12526 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), tuple_12521)
        # Assigning a type to the variable 'tp' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'tp', for_loop_var_12526)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 119):
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_12527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 45), 'int')
        
        # Call to py_vq(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to tp(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'X' (line 119)
        X_12530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 30), 'X', False)
        # Processing the call keyword arguments (line 119)
        kwargs_12531 = {}
        # Getting the type of 'tp' (line 119)
        tp_12529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'tp', False)
        # Calling tp(args, kwargs) (line 119)
        tp_call_result_12532 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), tp_12529, *[X_12530], **kwargs_12531)
        
        
        # Call to tp(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'initc' (line 119)
        initc_12534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'initc', False)
        # Processing the call keyword arguments (line 119)
        kwargs_12535 = {}
        # Getting the type of 'tp' (line 119)
        tp_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'tp', False)
        # Calling tp(args, kwargs) (line 119)
        tp_call_result_12536 = invoke(stypy.reporting.localization.Localization(__file__, 119, 34), tp_12533, *[initc_12534], **kwargs_12535)
        
        # Processing the call keyword arguments (line 119)
        kwargs_12537 = {}
        # Getting the type of 'py_vq' (line 119)
        py_vq_12528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 119)
        py_vq_call_result_12538 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), py_vq_12528, *[tp_call_result_12532, tp_call_result_12536], **kwargs_12537)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___12539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), py_vq_call_result_12538, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_12540 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), getitem___12539, int_12527)
        
        # Assigning a type to the variable 'label1' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'label1', subscript_call_result_12540)
        
        # Call to assert_array_equal(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'label1' (line 120)
        label1_12542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'label1', False)
        # Getting the type of 'LABEL1' (line 120)
        LABEL1_12543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 39), 'LABEL1', False)
        # Processing the call keyword arguments (line 120)
        kwargs_12544 = {}
        # Getting the type of 'assert_array_equal' (line 120)
        assert_array_equal_12541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 120)
        assert_array_equal_call_result_12545 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), assert_array_equal_12541, *[label1_12542, LABEL1_12543], **kwargs_12544)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_py_vq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_py_vq' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_12546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12546)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_py_vq'
        return stypy_return_type_12546


    @norecursion
    def test_vq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vq'
        module_type_store = module_type_store.open_function_context('test_vq', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test_vq.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test_vq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test_vq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test_vq.__dict__.__setitem__('stypy_function_name', 'TestVq.test_vq')
        TestVq.test_vq.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test_vq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test_vq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test_vq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test_vq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test_vq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test_vq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test_vq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vq(...)' code ##################

        
        # Assigning a Call to a Name (line 123):
        
        # Assigning a Call to a Name (line 123):
        
        # Call to concatenate(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_12549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_12550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        
        # Obtaining the type of the subscript
        int_12551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'int')
        # Getting the type of 'X' (line 123)
        X_12552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___12553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 34), X_12552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_12554 = invoke(stypy.reporting.localization.Localization(__file__, 123, 34), getitem___12553, int_12551)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 33), list_12550, subscript_call_result_12554)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 32), list_12549, list_12550)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_12555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        
        # Obtaining the type of the subscript
        int_12556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 44), 'int')
        # Getting the type of 'X' (line 123)
        X_12557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 42), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___12558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 42), X_12557, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_12559 = invoke(stypy.reporting.localization.Localization(__file__, 123, 42), getitem___12558, int_12556)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 41), list_12555, subscript_call_result_12559)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 32), list_12549, list_12555)
        # Adding element type (line 123)
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_12560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        
        # Obtaining the type of the subscript
        int_12561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 52), 'int')
        # Getting the type of 'X' (line 123)
        X_12562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 50), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___12563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 50), X_12562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_12564 = invoke(stypy.reporting.localization.Localization(__file__, 123, 50), getitem___12563, int_12561)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 49), list_12560, subscript_call_result_12564)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 32), list_12549, list_12560)
        
        # Processing the call keyword arguments (line 123)
        kwargs_12565 = {}
        # Getting the type of 'np' (line 123)
        np_12547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 123)
        concatenate_12548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), np_12547, 'concatenate')
        # Calling concatenate(args, kwargs) (line 123)
        concatenate_call_result_12566 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), concatenate_12548, *[list_12549], **kwargs_12565)
        
        # Assigning a type to the variable 'initc' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'initc', concatenate_call_result_12566)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 124)
        tuple_12567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 124)
        # Adding element type (line 124)
        # Getting the type of 'np' (line 124)
        np_12568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'np')
        # Obtaining the member 'array' of a type (line 124)
        array_12569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 18), np_12568, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 18), tuple_12567, array_12569)
        # Adding element type (line 124)
        # Getting the type of 'np' (line 124)
        np_12570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 124)
        matrix_12571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 28), np_12570, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 18), tuple_12567, matrix_12571)
        
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), tuple_12567)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_12572 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), tuple_12567)
        # Assigning a type to the variable 'tp' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'tp', for_loop_var_12572)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 125):
        
        # Assigning a Subscript to a Name (line 125):
        
        # Obtaining the type of the subscript
        int_12573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 12), 'int')
        
        # Call to vq(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to tp(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'X' (line 125)
        X_12577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'X', False)
        # Processing the call keyword arguments (line 125)
        kwargs_12578 = {}
        # Getting the type of 'tp' (line 125)
        tp_12576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'tp', False)
        # Calling tp(args, kwargs) (line 125)
        tp_call_result_12579 = invoke(stypy.reporting.localization.Localization(__file__, 125, 34), tp_12576, *[X_12577], **kwargs_12578)
        
        
        # Call to tp(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'initc' (line 125)
        initc_12581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 44), 'initc', False)
        # Processing the call keyword arguments (line 125)
        kwargs_12582 = {}
        # Getting the type of 'tp' (line 125)
        tp_12580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'tp', False)
        # Calling tp(args, kwargs) (line 125)
        tp_call_result_12583 = invoke(stypy.reporting.localization.Localization(__file__, 125, 41), tp_12580, *[initc_12581], **kwargs_12582)
        
        # Processing the call keyword arguments (line 125)
        kwargs_12584 = {}
        # Getting the type of '_vq' (line 125)
        _vq_12574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), '_vq', False)
        # Obtaining the member 'vq' of a type (line 125)
        vq_12575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), _vq_12574, 'vq')
        # Calling vq(args, kwargs) (line 125)
        vq_call_result_12585 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), vq_12575, *[tp_call_result_12579, tp_call_result_12583], **kwargs_12584)
        
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___12586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), vq_call_result_12585, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_12587 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), getitem___12586, int_12573)
        
        # Assigning a type to the variable 'tuple_var_assignment_11787' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'tuple_var_assignment_11787', subscript_call_result_12587)
        
        # Assigning a Subscript to a Name (line 125):
        
        # Obtaining the type of the subscript
        int_12588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 12), 'int')
        
        # Call to vq(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to tp(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'X' (line 125)
        X_12592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'X', False)
        # Processing the call keyword arguments (line 125)
        kwargs_12593 = {}
        # Getting the type of 'tp' (line 125)
        tp_12591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'tp', False)
        # Calling tp(args, kwargs) (line 125)
        tp_call_result_12594 = invoke(stypy.reporting.localization.Localization(__file__, 125, 34), tp_12591, *[X_12592], **kwargs_12593)
        
        
        # Call to tp(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'initc' (line 125)
        initc_12596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 44), 'initc', False)
        # Processing the call keyword arguments (line 125)
        kwargs_12597 = {}
        # Getting the type of 'tp' (line 125)
        tp_12595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'tp', False)
        # Calling tp(args, kwargs) (line 125)
        tp_call_result_12598 = invoke(stypy.reporting.localization.Localization(__file__, 125, 41), tp_12595, *[initc_12596], **kwargs_12597)
        
        # Processing the call keyword arguments (line 125)
        kwargs_12599 = {}
        # Getting the type of '_vq' (line 125)
        _vq_12589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), '_vq', False)
        # Obtaining the member 'vq' of a type (line 125)
        vq_12590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), _vq_12589, 'vq')
        # Calling vq(args, kwargs) (line 125)
        vq_call_result_12600 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), vq_12590, *[tp_call_result_12594, tp_call_result_12598], **kwargs_12599)
        
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___12601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), vq_call_result_12600, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_12602 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), getitem___12601, int_12588)
        
        # Assigning a type to the variable 'tuple_var_assignment_11788' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'tuple_var_assignment_11788', subscript_call_result_12602)
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'tuple_var_assignment_11787' (line 125)
        tuple_var_assignment_11787_12603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'tuple_var_assignment_11787')
        # Assigning a type to the variable 'label1' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'label1', tuple_var_assignment_11787_12603)
        
        # Assigning a Name to a Name (line 125):
        # Getting the type of 'tuple_var_assignment_11788' (line 125)
        tuple_var_assignment_11788_12604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'tuple_var_assignment_11788')
        # Assigning a type to the variable 'dist' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'dist', tuple_var_assignment_11788_12604)
        
        # Call to assert_array_equal(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'label1' (line 126)
        label1_12606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'label1', False)
        # Getting the type of 'LABEL1' (line 126)
        LABEL1_12607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'LABEL1', False)
        # Processing the call keyword arguments (line 126)
        kwargs_12608 = {}
        # Getting the type of 'assert_array_equal' (line 126)
        assert_array_equal_12605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 126)
        assert_array_equal_call_result_12609 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), assert_array_equal_12605, *[label1_12606, LABEL1_12607], **kwargs_12608)
        
        
        # Assigning a Call to a Tuple (line 127):
        
        # Assigning a Subscript to a Name (line 127):
        
        # Obtaining the type of the subscript
        int_12610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
        
        # Call to vq(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to tp(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'X' (line 127)
        X_12613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'X', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12614 = {}
        # Getting the type of 'tp' (line 127)
        tp_12612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'tp', False)
        # Calling tp(args, kwargs) (line 127)
        tp_call_result_12615 = invoke(stypy.reporting.localization.Localization(__file__, 127, 32), tp_12612, *[X_12613], **kwargs_12614)
        
        
        # Call to tp(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'initc' (line 127)
        initc_12617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'initc', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12618 = {}
        # Getting the type of 'tp' (line 127)
        tp_12616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'tp', False)
        # Calling tp(args, kwargs) (line 127)
        tp_call_result_12619 = invoke(stypy.reporting.localization.Localization(__file__, 127, 39), tp_12616, *[initc_12617], **kwargs_12618)
        
        # Processing the call keyword arguments (line 127)
        kwargs_12620 = {}
        # Getting the type of 'vq' (line 127)
        vq_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'vq', False)
        # Calling vq(args, kwargs) (line 127)
        vq_call_result_12621 = invoke(stypy.reporting.localization.Localization(__file__, 127, 29), vq_12611, *[tp_call_result_12615, tp_call_result_12619], **kwargs_12620)
        
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___12622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), vq_call_result_12621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_12623 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), getitem___12622, int_12610)
        
        # Assigning a type to the variable 'tuple_var_assignment_11789' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_11789', subscript_call_result_12623)
        
        # Assigning a Subscript to a Name (line 127):
        
        # Obtaining the type of the subscript
        int_12624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 12), 'int')
        
        # Call to vq(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Call to tp(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'X' (line 127)
        X_12627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'X', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12628 = {}
        # Getting the type of 'tp' (line 127)
        tp_12626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'tp', False)
        # Calling tp(args, kwargs) (line 127)
        tp_call_result_12629 = invoke(stypy.reporting.localization.Localization(__file__, 127, 32), tp_12626, *[X_12627], **kwargs_12628)
        
        
        # Call to tp(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'initc' (line 127)
        initc_12631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 42), 'initc', False)
        # Processing the call keyword arguments (line 127)
        kwargs_12632 = {}
        # Getting the type of 'tp' (line 127)
        tp_12630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'tp', False)
        # Calling tp(args, kwargs) (line 127)
        tp_call_result_12633 = invoke(stypy.reporting.localization.Localization(__file__, 127, 39), tp_12630, *[initc_12631], **kwargs_12632)
        
        # Processing the call keyword arguments (line 127)
        kwargs_12634 = {}
        # Getting the type of 'vq' (line 127)
        vq_12625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'vq', False)
        # Calling vq(args, kwargs) (line 127)
        vq_call_result_12635 = invoke(stypy.reporting.localization.Localization(__file__, 127, 29), vq_12625, *[tp_call_result_12629, tp_call_result_12633], **kwargs_12634)
        
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___12636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), vq_call_result_12635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_12637 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), getitem___12636, int_12624)
        
        # Assigning a type to the variable 'tuple_var_assignment_11790' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_11790', subscript_call_result_12637)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'tuple_var_assignment_11789' (line 127)
        tuple_var_assignment_11789_12638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_11789')
        # Assigning a type to the variable 'tlabel1' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tlabel1', tuple_var_assignment_11789_12638)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'tuple_var_assignment_11790' (line 127)
        tuple_var_assignment_11790_12639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'tuple_var_assignment_11790')
        # Assigning a type to the variable 'tdist' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'tdist', tuple_var_assignment_11790_12639)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_vq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vq' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_12640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vq'
        return stypy_return_type_12640


    @norecursion
    def test_vq_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vq_1d'
        module_type_store = module_type_store.open_function_context('test_vq_1d', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_function_name', 'TestVq.test_vq_1d')
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test_vq_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test_vq_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vq_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vq_1d(...)' code ##################

        
        # Assigning a Subscript to a Name (line 131):
        
        # Assigning a Subscript to a Name (line 131):
        
        # Obtaining the type of the subscript
        slice_12641 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 15), None, None, None)
        int_12642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 20), 'int')
        # Getting the type of 'X' (line 131)
        X_12643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'X')
        # Obtaining the member '__getitem__' of a type (line 131)
        getitem___12644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 15), X_12643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
        subscript_call_result_12645 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), getitem___12644, (slice_12641, int_12642))
        
        # Assigning a type to the variable 'data' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'data', subscript_call_result_12645)
        
        # Assigning a Subscript to a Name (line 132):
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        int_12646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 22), 'int')
        slice_12647 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 132, 16), None, int_12646, None)
        # Getting the type of 'data' (line 132)
        data_12648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___12649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), data_12648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_12650 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), getitem___12649, slice_12647)
        
        # Assigning a type to the variable 'initc' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'initc', subscript_call_result_12650)
        
        # Assigning a Call to a Tuple (line 133):
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_12651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to vq(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'data' (line 133)
        data_12654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'data', False)
        # Getting the type of 'initc' (line 133)
        initc_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'initc', False)
        # Processing the call keyword arguments (line 133)
        kwargs_12656 = {}
        # Getting the type of '_vq' (line 133)
        _vq_12652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), '_vq', False)
        # Obtaining the member 'vq' of a type (line 133)
        vq_12653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), _vq_12652, 'vq')
        # Calling vq(args, kwargs) (line 133)
        vq_call_result_12657 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), vq_12653, *[data_12654, initc_12655], **kwargs_12656)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___12658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), vq_call_result_12657, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_12659 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___12658, int_12651)
        
        # Assigning a type to the variable 'tuple_var_assignment_11791' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_11791', subscript_call_result_12659)
        
        # Assigning a Subscript to a Name (line 133):
        
        # Obtaining the type of the subscript
        int_12660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 8), 'int')
        
        # Call to vq(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'data' (line 133)
        data_12663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'data', False)
        # Getting the type of 'initc' (line 133)
        initc_12664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'initc', False)
        # Processing the call keyword arguments (line 133)
        kwargs_12665 = {}
        # Getting the type of '_vq' (line 133)
        _vq_12661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), '_vq', False)
        # Obtaining the member 'vq' of a type (line 133)
        vq_12662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 15), _vq_12661, 'vq')
        # Calling vq(args, kwargs) (line 133)
        vq_call_result_12666 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), vq_12662, *[data_12663, initc_12664], **kwargs_12665)
        
        # Obtaining the member '__getitem__' of a type (line 133)
        getitem___12667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), vq_call_result_12666, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 133)
        subscript_call_result_12668 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), getitem___12667, int_12660)
        
        # Assigning a type to the variable 'tuple_var_assignment_11792' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_11792', subscript_call_result_12668)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_11791' (line 133)
        tuple_var_assignment_11791_12669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_11791')
        # Assigning a type to the variable 'a' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'a', tuple_var_assignment_11791_12669)
        
        # Assigning a Name to a Name (line 133):
        # Getting the type of 'tuple_var_assignment_11792' (line 133)
        tuple_var_assignment_11792_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'tuple_var_assignment_11792')
        # Assigning a type to the variable 'b' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'b', tuple_var_assignment_11792_12670)
        
        # Assigning a Call to a Tuple (line 134):
        
        # Assigning a Subscript to a Name (line 134):
        
        # Obtaining the type of the subscript
        int_12671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
        
        # Call to py_vq(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining the type of the subscript
        slice_12673 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 23), None, None, None)
        # Getting the type of 'np' (line 134)
        np_12674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 134)
        newaxis_12675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 31), np_12674, 'newaxis')
        # Getting the type of 'data' (line 134)
        data_12676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), data_12676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), getitem___12677, (slice_12673, newaxis_12675))
        
        
        # Obtaining the type of the subscript
        slice_12679 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 44), None, None, None)
        # Getting the type of 'np' (line 134)
        np_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 53), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 134)
        newaxis_12681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 53), np_12680, 'newaxis')
        # Getting the type of 'initc' (line 134)
        initc_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'initc', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 44), initc_12682, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12684 = invoke(stypy.reporting.localization.Localization(__file__, 134, 44), getitem___12683, (slice_12679, newaxis_12681))
        
        # Processing the call keyword arguments (line 134)
        kwargs_12685 = {}
        # Getting the type of 'py_vq' (line 134)
        py_vq_12672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 134)
        py_vq_call_result_12686 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), py_vq_12672, *[subscript_call_result_12678, subscript_call_result_12684], **kwargs_12685)
        
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), py_vq_call_result_12686, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12688 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), getitem___12687, int_12671)
        
        # Assigning a type to the variable 'tuple_var_assignment_11793' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_11793', subscript_call_result_12688)
        
        # Assigning a Subscript to a Name (line 134):
        
        # Obtaining the type of the subscript
        int_12689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 8), 'int')
        
        # Call to py_vq(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Obtaining the type of the subscript
        slice_12691 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 23), None, None, None)
        # Getting the type of 'np' (line 134)
        np_12692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 134)
        newaxis_12693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 31), np_12692, 'newaxis')
        # Getting the type of 'data' (line 134)
        data_12694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), data_12694, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12696 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), getitem___12695, (slice_12691, newaxis_12693))
        
        
        # Obtaining the type of the subscript
        slice_12697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 134, 44), None, None, None)
        # Getting the type of 'np' (line 134)
        np_12698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 53), 'np', False)
        # Obtaining the member 'newaxis' of a type (line 134)
        newaxis_12699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 53), np_12698, 'newaxis')
        # Getting the type of 'initc' (line 134)
        initc_12700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'initc', False)
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 44), initc_12700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12702 = invoke(stypy.reporting.localization.Localization(__file__, 134, 44), getitem___12701, (slice_12697, newaxis_12699))
        
        # Processing the call keyword arguments (line 134)
        kwargs_12703 = {}
        # Getting the type of 'py_vq' (line 134)
        py_vq_12690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 134)
        py_vq_call_result_12704 = invoke(stypy.reporting.localization.Localization(__file__, 134, 17), py_vq_12690, *[subscript_call_result_12696, subscript_call_result_12702], **kwargs_12703)
        
        # Obtaining the member '__getitem__' of a type (line 134)
        getitem___12705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), py_vq_call_result_12704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 134)
        subscript_call_result_12706 = invoke(stypy.reporting.localization.Localization(__file__, 134, 8), getitem___12705, int_12689)
        
        # Assigning a type to the variable 'tuple_var_assignment_11794' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_11794', subscript_call_result_12706)
        
        # Assigning a Name to a Name (line 134):
        # Getting the type of 'tuple_var_assignment_11793' (line 134)
        tuple_var_assignment_11793_12707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_11793')
        # Assigning a type to the variable 'ta' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'ta', tuple_var_assignment_11793_12707)
        
        # Assigning a Name to a Name (line 134):
        # Getting the type of 'tuple_var_assignment_11794' (line 134)
        tuple_var_assignment_11794_12708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'tuple_var_assignment_11794')
        # Assigning a type to the variable 'tb' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'tb', tuple_var_assignment_11794_12708)
        
        # Call to assert_array_equal(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'a' (line 135)
        a_12710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'a', False)
        # Getting the type of 'ta' (line 135)
        ta_12711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'ta', False)
        # Processing the call keyword arguments (line 135)
        kwargs_12712 = {}
        # Getting the type of 'assert_array_equal' (line 135)
        assert_array_equal_12709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 135)
        assert_array_equal_call_result_12713 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), assert_array_equal_12709, *[a_12710, ta_12711], **kwargs_12712)
        
        
        # Call to assert_array_equal(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'b' (line 136)
        b_12715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'b', False)
        # Getting the type of 'tb' (line 136)
        tb_12716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), 'tb', False)
        # Processing the call keyword arguments (line 136)
        kwargs_12717 = {}
        # Getting the type of 'assert_array_equal' (line 136)
        assert_array_equal_12714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 136)
        assert_array_equal_call_result_12718 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), assert_array_equal_12714, *[b_12715, tb_12716], **kwargs_12717)
        
        
        # ################# End of 'test_vq_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vq_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_12719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vq_1d'
        return stypy_return_type_12719


    @norecursion
    def test__vq_sametype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__vq_sametype'
        module_type_store = module_type_store.open_function_context('test__vq_sametype', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_function_name', 'TestVq.test__vq_sametype')
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test__vq_sametype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test__vq_sametype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__vq_sametype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__vq_sametype(...)' code ##################

        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to array(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_12722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        float_12723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), list_12722, float_12723)
        # Adding element type (line 139)
        float_12724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), list_12722, float_12724)
        
        # Processing the call keyword arguments (line 139)
        # Getting the type of 'np' (line 139)
        np_12725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 39), 'np', False)
        # Obtaining the member 'float64' of a type (line 139)
        float64_12726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 39), np_12725, 'float64')
        keyword_12727 = float64_12726
        kwargs_12728 = {'dtype': keyword_12727}
        # Getting the type of 'np' (line 139)
        np_12720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 139)
        array_12721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), np_12720, 'array')
        # Calling array(args, kwargs) (line 139)
        array_call_result_12729 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), array_12721, *[list_12722], **kwargs_12728)
        
        # Assigning a type to the variable 'a' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'a', array_call_result_12729)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to astype(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'np' (line 140)
        np_12732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'np', False)
        # Obtaining the member 'float32' of a type (line 140)
        float32_12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), np_12732, 'float32')
        # Processing the call keyword arguments (line 140)
        kwargs_12734 = {}
        # Getting the type of 'a' (line 140)
        a_12730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'a', False)
        # Obtaining the member 'astype' of a type (line 140)
        astype_12731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), a_12730, 'astype')
        # Calling astype(args, kwargs) (line 140)
        astype_call_result_12735 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), astype_12731, *[float32_12733], **kwargs_12734)
        
        # Assigning a type to the variable 'b' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'b', astype_call_result_12735)
        
        # Call to assert_raises(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'TypeError' (line 141)
        TypeError_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'TypeError', False)
        # Getting the type of '_vq' (line 141)
        _vq_12738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), '_vq', False)
        # Obtaining the member 'vq' of a type (line 141)
        vq_12739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 33), _vq_12738, 'vq')
        # Getting the type of 'a' (line 141)
        a_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 41), 'a', False)
        # Getting the type of 'b' (line 141)
        b_12741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 44), 'b', False)
        # Processing the call keyword arguments (line 141)
        kwargs_12742 = {}
        # Getting the type of 'assert_raises' (line 141)
        assert_raises_12736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 141)
        assert_raises_call_result_12743 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), assert_raises_12736, *[TypeError_12737, vq_12739, a_12740, b_12741], **kwargs_12742)
        
        
        # ################# End of 'test__vq_sametype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__vq_sametype' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_12744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12744)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__vq_sametype'
        return stypy_return_type_12744


    @norecursion
    def test__vq_invalid_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test__vq_invalid_type'
        module_type_store = module_type_store.open_function_context('test__vq_invalid_type', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_function_name', 'TestVq.test__vq_invalid_type')
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test__vq_invalid_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test__vq_invalid_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test__vq_invalid_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test__vq_invalid_type(...)' code ##################

        
        # Assigning a Call to a Name (line 144):
        
        # Assigning a Call to a Name (line 144):
        
        # Call to array(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_12747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        int_12748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), list_12747, int_12748)
        # Adding element type (line 144)
        int_12749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), list_12747, int_12749)
        
        # Processing the call keyword arguments (line 144)
        # Getting the type of 'int' (line 144)
        int_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 35), 'int', False)
        keyword_12751 = int_12750
        kwargs_12752 = {'dtype': keyword_12751}
        # Getting the type of 'np' (line 144)
        np_12745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 144)
        array_12746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), np_12745, 'array')
        # Calling array(args, kwargs) (line 144)
        array_call_result_12753 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), array_12746, *[list_12747], **kwargs_12752)
        
        # Assigning a type to the variable 'a' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'a', array_call_result_12753)
        
        # Call to assert_raises(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'TypeError' (line 145)
        TypeError_12755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'TypeError', False)
        # Getting the type of '_vq' (line 145)
        _vq_12756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), '_vq', False)
        # Obtaining the member 'vq' of a type (line 145)
        vq_12757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 33), _vq_12756, 'vq')
        # Getting the type of 'a' (line 145)
        a_12758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 41), 'a', False)
        # Getting the type of 'a' (line 145)
        a_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 44), 'a', False)
        # Processing the call keyword arguments (line 145)
        kwargs_12760 = {}
        # Getting the type of 'assert_raises' (line 145)
        assert_raises_12754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 145)
        assert_raises_call_result_12761 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assert_raises_12754, *[TypeError_12755, vq_12757, a_12758, a_12759], **kwargs_12760)
        
        
        # ################# End of 'test__vq_invalid_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test__vq_invalid_type' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_12762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12762)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test__vq_invalid_type'
        return stypy_return_type_12762


    @norecursion
    def test_vq_large_nfeat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vq_large_nfeat'
        module_type_store = module_type_store.open_function_context('test_vq_large_nfeat', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_function_name', 'TestVq.test_vq_large_nfeat')
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test_vq_large_nfeat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test_vq_large_nfeat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vq_large_nfeat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vq_large_nfeat(...)' code ##################

        
        # Assigning a Call to a Name (line 148):
        
        # Assigning a Call to a Name (line 148):
        
        # Call to rand(...): (line 148)
        # Processing the call arguments (line 148)
        int_12766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 27), 'int')
        int_12767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 31), 'int')
        # Processing the call keyword arguments (line 148)
        kwargs_12768 = {}
        # Getting the type of 'np' (line 148)
        np_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 148)
        random_12764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), np_12763, 'random')
        # Obtaining the member 'rand' of a type (line 148)
        rand_12765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), random_12764, 'rand')
        # Calling rand(args, kwargs) (line 148)
        rand_call_result_12769 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), rand_12765, *[int_12766, int_12767], **kwargs_12768)
        
        # Assigning a type to the variable 'X' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'X', rand_call_result_12769)
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to rand(...): (line 149)
        # Processing the call arguments (line 149)
        int_12773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 35), 'int')
        int_12774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 38), 'int')
        # Processing the call keyword arguments (line 149)
        kwargs_12775 = {}
        # Getting the type of 'np' (line 149)
        np_12770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 149)
        random_12771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), np_12770, 'random')
        # Obtaining the member 'rand' of a type (line 149)
        rand_12772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), random_12771, 'rand')
        # Calling rand(args, kwargs) (line 149)
        rand_call_result_12776 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), rand_12772, *[int_12773, int_12774], **kwargs_12775)
        
        # Assigning a type to the variable 'code_book' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'code_book', rand_call_result_12776)
        
        # Assigning a Call to a Tuple (line 151):
        
        # Assigning a Subscript to a Name (line 151):
        
        # Obtaining the type of the subscript
        int_12777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        
        # Call to vq(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'X' (line 151)
        X_12780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'X', False)
        # Getting the type of 'code_book' (line 151)
        code_book_12781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'code_book', False)
        # Processing the call keyword arguments (line 151)
        kwargs_12782 = {}
        # Getting the type of '_vq' (line 151)
        _vq_12778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 151)
        vq_12779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 23), _vq_12778, 'vq')
        # Calling vq(args, kwargs) (line 151)
        vq_call_result_12783 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), vq_12779, *[X_12780, code_book_12781], **kwargs_12782)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___12784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), vq_call_result_12783, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_12785 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___12784, int_12777)
        
        # Assigning a type to the variable 'tuple_var_assignment_11795' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_11795', subscript_call_result_12785)
        
        # Assigning a Subscript to a Name (line 151):
        
        # Obtaining the type of the subscript
        int_12786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'int')
        
        # Call to vq(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'X' (line 151)
        X_12789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 30), 'X', False)
        # Getting the type of 'code_book' (line 151)
        code_book_12790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'code_book', False)
        # Processing the call keyword arguments (line 151)
        kwargs_12791 = {}
        # Getting the type of '_vq' (line 151)
        _vq_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 151)
        vq_12788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 23), _vq_12787, 'vq')
        # Calling vq(args, kwargs) (line 151)
        vq_call_result_12792 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), vq_12788, *[X_12789, code_book_12790], **kwargs_12791)
        
        # Obtaining the member '__getitem__' of a type (line 151)
        getitem___12793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), vq_call_result_12792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 151)
        subscript_call_result_12794 = invoke(stypy.reporting.localization.Localization(__file__, 151, 8), getitem___12793, int_12786)
        
        # Assigning a type to the variable 'tuple_var_assignment_11796' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_11796', subscript_call_result_12794)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_var_assignment_11795' (line 151)
        tuple_var_assignment_11795_12795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_11795')
        # Assigning a type to the variable 'codes0' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'codes0', tuple_var_assignment_11795_12795)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_var_assignment_11796' (line 151)
        tuple_var_assignment_11796_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_var_assignment_11796')
        # Assigning a type to the variable 'dis0' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'dis0', tuple_var_assignment_11796_12796)
        
        # Assigning a Call to a Tuple (line 152):
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_12797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'int')
        
        # Call to py_vq(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'X' (line 152)
        X_12799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'X', False)
        # Getting the type of 'code_book' (line 152)
        code_book_12800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'code_book', False)
        # Processing the call keyword arguments (line 152)
        kwargs_12801 = {}
        # Getting the type of 'py_vq' (line 152)
        py_vq_12798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 152)
        py_vq_call_result_12802 = invoke(stypy.reporting.localization.Localization(__file__, 152, 23), py_vq_12798, *[X_12799, code_book_12800], **kwargs_12801)
        
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___12803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), py_vq_call_result_12802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_12804 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), getitem___12803, int_12797)
        
        # Assigning a type to the variable 'tuple_var_assignment_11797' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_11797', subscript_call_result_12804)
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_12805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'int')
        
        # Call to py_vq(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'X' (line 152)
        X_12807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'X', False)
        # Getting the type of 'code_book' (line 152)
        code_book_12808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'code_book', False)
        # Processing the call keyword arguments (line 152)
        kwargs_12809 = {}
        # Getting the type of 'py_vq' (line 152)
        py_vq_12806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 152)
        py_vq_call_result_12810 = invoke(stypy.reporting.localization.Localization(__file__, 152, 23), py_vq_12806, *[X_12807, code_book_12808], **kwargs_12809)
        
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___12811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), py_vq_call_result_12810, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_12812 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), getitem___12811, int_12805)
        
        # Assigning a type to the variable 'tuple_var_assignment_11798' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_11798', subscript_call_result_12812)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_11797' (line 152)
        tuple_var_assignment_11797_12813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_11797')
        # Assigning a type to the variable 'codes1' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'codes1', tuple_var_assignment_11797_12813)
        
        # Assigning a Name to a Name (line 152):
        # Getting the type of 'tuple_var_assignment_11798' (line 152)
        tuple_var_assignment_11798_12814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'tuple_var_assignment_11798')
        # Assigning a type to the variable 'dis1' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'dis1', tuple_var_assignment_11798_12814)
        
        # Call to assert_allclose(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'dis0' (line 153)
        dis0_12816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'dis0', False)
        # Getting the type of 'dis1' (line 153)
        dis1_12817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'dis1', False)
        float_12818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 36), 'float')
        # Processing the call keyword arguments (line 153)
        kwargs_12819 = {}
        # Getting the type of 'assert_allclose' (line 153)
        assert_allclose_12815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 153)
        assert_allclose_call_result_12820 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_allclose_12815, *[dis0_12816, dis1_12817, float_12818], **kwargs_12819)
        
        
        # Call to assert_array_equal(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'codes0' (line 154)
        codes0_12822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 27), 'codes0', False)
        # Getting the type of 'codes1' (line 154)
        codes1_12823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'codes1', False)
        # Processing the call keyword arguments (line 154)
        kwargs_12824 = {}
        # Getting the type of 'assert_array_equal' (line 154)
        assert_array_equal_12821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 154)
        assert_array_equal_call_result_12825 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assert_array_equal_12821, *[codes0_12822, codes1_12823], **kwargs_12824)
        
        
        # Assigning a Call to a Name (line 156):
        
        # Assigning a Call to a Name (line 156):
        
        # Call to astype(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'np' (line 156)
        np_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'np', False)
        # Obtaining the member 'float32' of a type (line 156)
        float32_12829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 21), np_12828, 'float32')
        # Processing the call keyword arguments (line 156)
        kwargs_12830 = {}
        # Getting the type of 'X' (line 156)
        X_12826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'X', False)
        # Obtaining the member 'astype' of a type (line 156)
        astype_12827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), X_12826, 'astype')
        # Calling astype(args, kwargs) (line 156)
        astype_call_result_12831 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), astype_12827, *[float32_12829], **kwargs_12830)
        
        # Assigning a type to the variable 'X' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'X', astype_call_result_12831)
        
        # Assigning a Call to a Name (line 157):
        
        # Assigning a Call to a Name (line 157):
        
        # Call to astype(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'np' (line 157)
        np_12834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'np', False)
        # Obtaining the member 'float32' of a type (line 157)
        float32_12835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 37), np_12834, 'float32')
        # Processing the call keyword arguments (line 157)
        kwargs_12836 = {}
        # Getting the type of 'code_book' (line 157)
        code_book_12832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'code_book', False)
        # Obtaining the member 'astype' of a type (line 157)
        astype_12833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 20), code_book_12832, 'astype')
        # Calling astype(args, kwargs) (line 157)
        astype_call_result_12837 = invoke(stypy.reporting.localization.Localization(__file__, 157, 20), astype_12833, *[float32_12835], **kwargs_12836)
        
        # Assigning a type to the variable 'code_book' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'code_book', astype_call_result_12837)
        
        # Assigning a Call to a Tuple (line 159):
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_12838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        
        # Call to vq(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'X' (line 159)
        X_12841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'X', False)
        # Getting the type of 'code_book' (line 159)
        code_book_12842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'code_book', False)
        # Processing the call keyword arguments (line 159)
        kwargs_12843 = {}
        # Getting the type of '_vq' (line 159)
        _vq_12839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 159)
        vq_12840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), _vq_12839, 'vq')
        # Calling vq(args, kwargs) (line 159)
        vq_call_result_12844 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), vq_12840, *[X_12841, code_book_12842], **kwargs_12843)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___12845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), vq_call_result_12844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_12846 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), getitem___12845, int_12838)
        
        # Assigning a type to the variable 'tuple_var_assignment_11799' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_11799', subscript_call_result_12846)
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        int_12847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 8), 'int')
        
        # Call to vq(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'X' (line 159)
        X_12850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'X', False)
        # Getting the type of 'code_book' (line 159)
        code_book_12851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 33), 'code_book', False)
        # Processing the call keyword arguments (line 159)
        kwargs_12852 = {}
        # Getting the type of '_vq' (line 159)
        _vq_12848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 159)
        vq_12849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), _vq_12848, 'vq')
        # Calling vq(args, kwargs) (line 159)
        vq_call_result_12853 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), vq_12849, *[X_12850, code_book_12851], **kwargs_12852)
        
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___12854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), vq_call_result_12853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_12855 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), getitem___12854, int_12847)
        
        # Assigning a type to the variable 'tuple_var_assignment_11800' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_11800', subscript_call_result_12855)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_11799' (line 159)
        tuple_var_assignment_11799_12856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_11799')
        # Assigning a type to the variable 'codes0' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'codes0', tuple_var_assignment_11799_12856)
        
        # Assigning a Name to a Name (line 159):
        # Getting the type of 'tuple_var_assignment_11800' (line 159)
        tuple_var_assignment_11800_12857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'tuple_var_assignment_11800')
        # Assigning a type to the variable 'dis0' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'dis0', tuple_var_assignment_11800_12857)
        
        # Assigning a Call to a Tuple (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_12858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to py_vq(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'X' (line 160)
        X_12860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'X', False)
        # Getting the type of 'code_book' (line 160)
        code_book_12861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'code_book', False)
        # Processing the call keyword arguments (line 160)
        kwargs_12862 = {}
        # Getting the type of 'py_vq' (line 160)
        py_vq_12859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 160)
        py_vq_call_result_12863 = invoke(stypy.reporting.localization.Localization(__file__, 160, 23), py_vq_12859, *[X_12860, code_book_12861], **kwargs_12862)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___12864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), py_vq_call_result_12863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_12865 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___12864, int_12858)
        
        # Assigning a type to the variable 'tuple_var_assignment_11801' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_11801', subscript_call_result_12865)
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_12866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to py_vq(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'X' (line 160)
        X_12868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'X', False)
        # Getting the type of 'code_book' (line 160)
        code_book_12869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'code_book', False)
        # Processing the call keyword arguments (line 160)
        kwargs_12870 = {}
        # Getting the type of 'py_vq' (line 160)
        py_vq_12867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 160)
        py_vq_call_result_12871 = invoke(stypy.reporting.localization.Localization(__file__, 160, 23), py_vq_12867, *[X_12868, code_book_12869], **kwargs_12870)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___12872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), py_vq_call_result_12871, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_12873 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___12872, int_12866)
        
        # Assigning a type to the variable 'tuple_var_assignment_11802' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_11802', subscript_call_result_12873)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_11801' (line 160)
        tuple_var_assignment_11801_12874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_11801')
        # Assigning a type to the variable 'codes1' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'codes1', tuple_var_assignment_11801_12874)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_11802' (line 160)
        tuple_var_assignment_11802_12875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_11802')
        # Assigning a type to the variable 'dis1' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'dis1', tuple_var_assignment_11802_12875)
        
        # Call to assert_allclose(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'dis0' (line 161)
        dis0_12877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'dis0', False)
        # Getting the type of 'dis1' (line 161)
        dis1_12878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 30), 'dis1', False)
        float_12879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 36), 'float')
        # Processing the call keyword arguments (line 161)
        kwargs_12880 = {}
        # Getting the type of 'assert_allclose' (line 161)
        assert_allclose_12876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 161)
        assert_allclose_call_result_12881 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_allclose_12876, *[dis0_12877, dis1_12878, float_12879], **kwargs_12880)
        
        
        # Call to assert_array_equal(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'codes0' (line 162)
        codes0_12883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'codes0', False)
        # Getting the type of 'codes1' (line 162)
        codes1_12884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'codes1', False)
        # Processing the call keyword arguments (line 162)
        kwargs_12885 = {}
        # Getting the type of 'assert_array_equal' (line 162)
        assert_array_equal_12882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 162)
        assert_array_equal_call_result_12886 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), assert_array_equal_12882, *[codes0_12883, codes1_12884], **kwargs_12885)
        
        
        # ################# End of 'test_vq_large_nfeat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vq_large_nfeat' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_12887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12887)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vq_large_nfeat'
        return stypy_return_type_12887


    @norecursion
    def test_vq_large_features(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_vq_large_features'
        module_type_store = module_type_store.open_function_context('test_vq_large_features', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_localization', localization)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_function_name', 'TestVq.test_vq_large_features')
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_param_names_list', [])
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestVq.test_vq_large_features.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.test_vq_large_features', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_vq_large_features', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_vq_large_features(...)' code ##################

        
        # Assigning a BinOp to a Name (line 165):
        
        # Assigning a BinOp to a Name (line 165):
        
        # Call to rand(...): (line 165)
        # Processing the call arguments (line 165)
        int_12891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 27), 'int')
        int_12892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 31), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_12893 = {}
        # Getting the type of 'np' (line 165)
        np_12888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 165)
        random_12889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), np_12888, 'random')
        # Obtaining the member 'rand' of a type (line 165)
        rand_12890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), random_12889, 'rand')
        # Calling rand(args, kwargs) (line 165)
        rand_call_result_12894 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), rand_12890, *[int_12891, int_12892], **kwargs_12893)
        
        int_12895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 36), 'int')
        # Applying the binary operator '*' (line 165)
        result_mul_12896 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 12), '*', rand_call_result_12894, int_12895)
        
        # Assigning a type to the variable 'X' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'X', result_mul_12896)
        
        # Assigning a BinOp to a Name (line 166):
        
        # Assigning a BinOp to a Name (line 166):
        
        # Call to rand(...): (line 166)
        # Processing the call arguments (line 166)
        int_12900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 35), 'int')
        int_12901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 38), 'int')
        # Processing the call keyword arguments (line 166)
        kwargs_12902 = {}
        # Getting the type of 'np' (line 166)
        np_12897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 166)
        random_12898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), np_12897, 'random')
        # Obtaining the member 'rand' of a type (line 166)
        rand_12899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), random_12898, 'rand')
        # Calling rand(args, kwargs) (line 166)
        rand_call_result_12903 = invoke(stypy.reporting.localization.Localization(__file__, 166, 20), rand_12899, *[int_12900, int_12901], **kwargs_12902)
        
        int_12904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 43), 'int')
        # Applying the binary operator '*' (line 166)
        result_mul_12905 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 20), '*', rand_call_result_12903, int_12904)
        
        # Assigning a type to the variable 'code_book' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'code_book', result_mul_12905)
        
        # Assigning a Call to a Tuple (line 168):
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_12906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to vq(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'X' (line 168)
        X_12909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'X', False)
        # Getting the type of 'code_book' (line 168)
        code_book_12910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'code_book', False)
        # Processing the call keyword arguments (line 168)
        kwargs_12911 = {}
        # Getting the type of '_vq' (line 168)
        _vq_12907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 168)
        vq_12908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 23), _vq_12907, 'vq')
        # Calling vq(args, kwargs) (line 168)
        vq_call_result_12912 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), vq_12908, *[X_12909, code_book_12910], **kwargs_12911)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___12913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), vq_call_result_12912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_12914 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___12913, int_12906)
        
        # Assigning a type to the variable 'tuple_var_assignment_11803' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_11803', subscript_call_result_12914)
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_12915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'int')
        
        # Call to vq(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'X' (line 168)
        X_12918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'X', False)
        # Getting the type of 'code_book' (line 168)
        code_book_12919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'code_book', False)
        # Processing the call keyword arguments (line 168)
        kwargs_12920 = {}
        # Getting the type of '_vq' (line 168)
        _vq_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), '_vq', False)
        # Obtaining the member 'vq' of a type (line 168)
        vq_12917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 23), _vq_12916, 'vq')
        # Calling vq(args, kwargs) (line 168)
        vq_call_result_12921 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), vq_12917, *[X_12918, code_book_12919], **kwargs_12920)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___12922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), vq_call_result_12921, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_12923 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), getitem___12922, int_12915)
        
        # Assigning a type to the variable 'tuple_var_assignment_11804' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_11804', subscript_call_result_12923)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_11803' (line 168)
        tuple_var_assignment_11803_12924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_11803')
        # Assigning a type to the variable 'codes0' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'codes0', tuple_var_assignment_11803_12924)
        
        # Assigning a Name to a Name (line 168):
        # Getting the type of 'tuple_var_assignment_11804' (line 168)
        tuple_var_assignment_11804_12925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'tuple_var_assignment_11804')
        # Assigning a type to the variable 'dis0' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'dis0', tuple_var_assignment_11804_12925)
        
        # Assigning a Call to a Tuple (line 169):
        
        # Assigning a Subscript to a Name (line 169):
        
        # Obtaining the type of the subscript
        int_12926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
        
        # Call to py_vq(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'X' (line 169)
        X_12928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'X', False)
        # Getting the type of 'code_book' (line 169)
        code_book_12929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'code_book', False)
        # Processing the call keyword arguments (line 169)
        kwargs_12930 = {}
        # Getting the type of 'py_vq' (line 169)
        py_vq_12927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 169)
        py_vq_call_result_12931 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), py_vq_12927, *[X_12928, code_book_12929], **kwargs_12930)
        
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___12932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), py_vq_call_result_12931, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_12933 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___12932, int_12926)
        
        # Assigning a type to the variable 'tuple_var_assignment_11805' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_11805', subscript_call_result_12933)
        
        # Assigning a Subscript to a Name (line 169):
        
        # Obtaining the type of the subscript
        int_12934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
        
        # Call to py_vq(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'X' (line 169)
        X_12936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'X', False)
        # Getting the type of 'code_book' (line 169)
        code_book_12937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'code_book', False)
        # Processing the call keyword arguments (line 169)
        kwargs_12938 = {}
        # Getting the type of 'py_vq' (line 169)
        py_vq_12935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'py_vq', False)
        # Calling py_vq(args, kwargs) (line 169)
        py_vq_call_result_12939 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), py_vq_12935, *[X_12936, code_book_12937], **kwargs_12938)
        
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___12940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), py_vq_call_result_12939, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_12941 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), getitem___12940, int_12934)
        
        # Assigning a type to the variable 'tuple_var_assignment_11806' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_11806', subscript_call_result_12941)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'tuple_var_assignment_11805' (line 169)
        tuple_var_assignment_11805_12942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_11805')
        # Assigning a type to the variable 'codes1' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'codes1', tuple_var_assignment_11805_12942)
        
        # Assigning a Name to a Name (line 169):
        # Getting the type of 'tuple_var_assignment_11806' (line 169)
        tuple_var_assignment_11806_12943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple_var_assignment_11806')
        # Assigning a type to the variable 'dis1' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'dis1', tuple_var_assignment_11806_12943)
        
        # Call to assert_allclose(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'dis0' (line 170)
        dis0_12945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'dis0', False)
        # Getting the type of 'dis1' (line 170)
        dis1_12946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'dis1', False)
        float_12947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 36), 'float')
        # Processing the call keyword arguments (line 170)
        kwargs_12948 = {}
        # Getting the type of 'assert_allclose' (line 170)
        assert_allclose_12944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 170)
        assert_allclose_call_result_12949 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), assert_allclose_12944, *[dis0_12945, dis1_12946, float_12947], **kwargs_12948)
        
        
        # Call to assert_array_equal(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'codes0' (line 171)
        codes0_12951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 27), 'codes0', False)
        # Getting the type of 'codes1' (line 171)
        codes1_12952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 35), 'codes1', False)
        # Processing the call keyword arguments (line 171)
        kwargs_12953 = {}
        # Getting the type of 'assert_array_equal' (line 171)
        assert_array_equal_12950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 171)
        assert_array_equal_call_result_12954 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_array_equal_12950, *[codes0_12951, codes1_12952], **kwargs_12953)
        
        
        # ################# End of 'test_vq_large_features(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_vq_large_features' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_12955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_vq_large_features'
        return stypy_return_type_12955


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 115, 0, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestVq.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestVq' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'TestVq', TestVq)
# Declaration of the 'TestKMean' class

class TestKMean(object, ):

    @norecursion
    def test_large_features(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_large_features'
        module_type_store = module_type_store.open_function_context('test_large_features', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_large_features.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_large_features')
        TestKMean.test_large_features.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_large_features.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_large_features.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_large_features', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_large_features', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_large_features(...)' code ##################

        
        # Assigning a Num to a Name (line 178):
        
        # Assigning a Num to a Name (line 178):
        int_12956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 12), 'int')
        # Assigning a type to the variable 'd' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'd', int_12956)
        
        # Assigning a Num to a Name (line 179):
        
        # Assigning a Num to a Name (line 179):
        int_12957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 12), 'int')
        # Assigning a type to the variable 'n' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'n', int_12957)
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to randn(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'd' (line 181)
        d_12961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'd', False)
        # Processing the call keyword arguments (line 181)
        kwargs_12962 = {}
        # Getting the type of 'np' (line 181)
        np_12958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 181)
        random_12959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 13), np_12958, 'random')
        # Obtaining the member 'randn' of a type (line 181)
        randn_12960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 13), random_12959, 'randn')
        # Calling randn(args, kwargs) (line 181)
        randn_call_result_12963 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), randn_12960, *[d_12961], **kwargs_12962)
        
        # Assigning a type to the variable 'm1' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'm1', randn_call_result_12963)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to randn(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'd' (line 182)
        d_12967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'd', False)
        # Processing the call keyword arguments (line 182)
        kwargs_12968 = {}
        # Getting the type of 'np' (line 182)
        np_12964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'np', False)
        # Obtaining the member 'random' of a type (line 182)
        random_12965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), np_12964, 'random')
        # Obtaining the member 'randn' of a type (line 182)
        randn_12966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), random_12965, 'randn')
        # Calling randn(args, kwargs) (line 182)
        randn_call_result_12969 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), randn_12966, *[d_12967], **kwargs_12968)
        
        # Assigning a type to the variable 'm2' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'm2', randn_call_result_12969)
        
        # Assigning a BinOp to a Name (line 183):
        
        # Assigning a BinOp to a Name (line 183):
        int_12970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'int')
        
        # Call to randn(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'n' (line 183)
        n_12974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 36), 'n', False)
        # Getting the type of 'd' (line 183)
        d_12975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'd', False)
        # Processing the call keyword arguments (line 183)
        kwargs_12976 = {}
        # Getting the type of 'np' (line 183)
        np_12971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 183)
        random_12972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), np_12971, 'random')
        # Obtaining the member 'randn' of a type (line 183)
        randn_12973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), random_12972, 'randn')
        # Calling randn(args, kwargs) (line 183)
        randn_call_result_12977 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), randn_12973, *[n_12974, d_12975], **kwargs_12976)
        
        # Applying the binary operator '*' (line 183)
        result_mul_12978 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 12), '*', int_12970, randn_call_result_12977)
        
        int_12979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 44), 'int')
        # Getting the type of 'm1' (line 183)
        m1_12980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 52), 'm1')
        # Applying the binary operator '*' (line 183)
        result_mul_12981 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 44), '*', int_12979, m1_12980)
        
        # Applying the binary operator '-' (line 183)
        result_sub_12982 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 12), '-', result_mul_12978, result_mul_12981)
        
        # Assigning a type to the variable 'x' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'x', result_sub_12982)
        
        # Assigning a BinOp to a Name (line 184):
        
        # Assigning a BinOp to a Name (line 184):
        int_12983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 12), 'int')
        
        # Call to randn(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'n' (line 184)
        n_12987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'n', False)
        # Getting the type of 'd' (line 184)
        d_12988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 39), 'd', False)
        # Processing the call keyword arguments (line 184)
        kwargs_12989 = {}
        # Getting the type of 'np' (line 184)
        np_12984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'np', False)
        # Obtaining the member 'random' of a type (line 184)
        random_12985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 20), np_12984, 'random')
        # Obtaining the member 'randn' of a type (line 184)
        randn_12986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 20), random_12985, 'randn')
        # Calling randn(args, kwargs) (line 184)
        randn_call_result_12990 = invoke(stypy.reporting.localization.Localization(__file__, 184, 20), randn_12986, *[n_12987, d_12988], **kwargs_12989)
        
        # Applying the binary operator '*' (line 184)
        result_mul_12991 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 12), '*', int_12983, randn_call_result_12990)
        
        int_12992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 44), 'int')
        # Getting the type of 'm2' (line 184)
        m2_12993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 52), 'm2')
        # Applying the binary operator '*' (line 184)
        result_mul_12994 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 44), '*', int_12992, m2_12993)
        
        # Applying the binary operator '+' (line 184)
        result_add_12995 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 12), '+', result_mul_12991, result_mul_12994)
        
        # Assigning a type to the variable 'y' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'y', result_add_12995)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to empty(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_12998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        
        # Obtaining the type of the subscript
        int_12999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'int')
        # Getting the type of 'x' (line 186)
        x_13000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'x', False)
        # Obtaining the member 'shape' of a type (line 186)
        shape_13001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), x_13000, 'shape')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___13002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), shape_13001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_13003 = invoke(stypy.reporting.localization.Localization(__file__, 186, 25), getitem___13002, int_12999)
        
        
        # Obtaining the type of the subscript
        int_13004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 46), 'int')
        # Getting the type of 'y' (line 186)
        y_13005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 38), 'y', False)
        # Obtaining the member 'shape' of a type (line 186)
        shape_13006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 38), y_13005, 'shape')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___13007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 38), shape_13006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_13008 = invoke(stypy.reporting.localization.Localization(__file__, 186, 38), getitem___13007, int_13004)
        
        # Applying the binary operator '+' (line 186)
        result_add_13009 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 25), '+', subscript_call_result_13003, subscript_call_result_13008)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 25), tuple_12998, result_add_13009)
        # Adding element type (line 186)
        # Getting the type of 'd' (line 186)
        d_13010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 50), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 25), tuple_12998, d_13010)
        
        # Getting the type of 'np' (line 186)
        np_13011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 54), 'np', False)
        # Obtaining the member 'double' of a type (line 186)
        double_13012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 54), np_13011, 'double')
        # Processing the call keyword arguments (line 186)
        kwargs_13013 = {}
        # Getting the type of 'np' (line 186)
        np_12996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 186)
        empty_12997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 15), np_12996, 'empty')
        # Calling empty(args, kwargs) (line 186)
        empty_call_result_13014 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), empty_12997, *[tuple_12998, double_13012], **kwargs_13013)
        
        # Assigning a type to the variable 'data' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'data', empty_call_result_13014)
        
        # Assigning a Name to a Subscript (line 187):
        
        # Assigning a Name to a Subscript (line 187):
        # Getting the type of 'x' (line 187)
        x_13015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'x')
        # Getting the type of 'data' (line 187)
        data_13016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'data')
        
        # Obtaining the type of the subscript
        int_13017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 22), 'int')
        # Getting the type of 'x' (line 187)
        x_13018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'x')
        # Obtaining the member 'shape' of a type (line 187)
        shape_13019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 14), x_13018, 'shape')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___13020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 14), shape_13019, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_13021 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), getitem___13020, int_13017)
        
        slice_13022 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 8), None, subscript_call_result_13021, None)
        # Storing an element on a container (line 187)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 8), data_13016, (slice_13022, x_13015))
        
        # Assigning a Name to a Subscript (line 188):
        
        # Assigning a Name to a Subscript (line 188):
        # Getting the type of 'y' (line 188)
        y_13023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'y')
        # Getting the type of 'data' (line 188)
        data_13024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'data')
        
        # Obtaining the type of the subscript
        int_13025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'int')
        # Getting the type of 'x' (line 188)
        x_13026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'x')
        # Obtaining the member 'shape' of a type (line 188)
        shape_13027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 13), x_13026, 'shape')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___13028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 13), shape_13027, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_13029 = invoke(stypy.reporting.localization.Localization(__file__, 188, 13), getitem___13028, int_13025)
        
        slice_13030 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 8), subscript_call_result_13029, None, None)
        # Storing an element on a container (line 188)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 8), data_13024, (slice_13030, y_13023))
        
        # Call to kmeans(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'data' (line 190)
        data_13032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'data', False)
        int_13033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 21), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_13034 = {}
        # Getting the type of 'kmeans' (line 190)
        kmeans_13031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'kmeans', False)
        # Calling kmeans(args, kwargs) (line 190)
        kmeans_call_result_13035 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), kmeans_13031, *[data_13032, int_13033], **kwargs_13034)
        
        
        # ################# End of 'test_large_features(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_large_features' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_13036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13036)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_large_features'
        return stypy_return_type_13036


    @norecursion
    def test_kmeans_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans_simple'
        module_type_store = module_type_store.open_function_context('test_kmeans_simple', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans_simple')
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans_simple(...)' code ##################

        
        # Call to seed(...): (line 193)
        # Processing the call arguments (line 193)
        int_13040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 23), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_13041 = {}
        # Getting the type of 'np' (line 193)
        np_13037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 193)
        random_13038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), np_13037, 'random')
        # Obtaining the member 'seed' of a type (line 193)
        seed_13039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), random_13038, 'seed')
        # Calling seed(args, kwargs) (line 193)
        seed_call_result_13042 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), seed_13039, *[int_13040], **kwargs_13041)
        
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to concatenate(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_13045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_13046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining the type of the subscript
        int_13047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'int')
        # Getting the type of 'X' (line 194)
        X_13048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___13049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 34), X_13048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_13050 = invoke(stypy.reporting.localization.Localization(__file__, 194, 34), getitem___13049, int_13047)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 33), list_13046, subscript_call_result_13050)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), list_13045, list_13046)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_13051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining the type of the subscript
        int_13052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 44), 'int')
        # Getting the type of 'X' (line 194)
        X_13053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 42), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___13054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 42), X_13053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_13055 = invoke(stypy.reporting.localization.Localization(__file__, 194, 42), getitem___13054, int_13052)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 41), list_13051, subscript_call_result_13055)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), list_13045, list_13051)
        # Adding element type (line 194)
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_13056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        # Adding element type (line 194)
        
        # Obtaining the type of the subscript
        int_13057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'int')
        # Getting the type of 'X' (line 194)
        X_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 50), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 194)
        getitem___13059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 50), X_13058, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 194)
        subscript_call_result_13060 = invoke(stypy.reporting.localization.Localization(__file__, 194, 50), getitem___13059, int_13057)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 49), list_13056, subscript_call_result_13060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 32), list_13045, list_13056)
        
        # Processing the call keyword arguments (line 194)
        kwargs_13061 = {}
        # Getting the type of 'np' (line 194)
        np_13043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 194)
        concatenate_13044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 16), np_13043, 'concatenate')
        # Calling concatenate(args, kwargs) (line 194)
        concatenate_call_result_13062 = invoke(stypy.reporting.localization.Localization(__file__, 194, 16), concatenate_13044, *[list_13045], **kwargs_13061)
        
        # Assigning a type to the variable 'initc' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'initc', concatenate_call_result_13062)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 195)
        tuple_13063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 195)
        # Adding element type (line 195)
        # Getting the type of 'np' (line 195)
        np_13064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'np')
        # Obtaining the member 'array' of a type (line 195)
        array_13065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 18), np_13064, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), tuple_13063, array_13065)
        # Adding element type (line 195)
        # Getting the type of 'np' (line 195)
        np_13066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 195)
        matrix_13067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 28), np_13066, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 18), tuple_13063, matrix_13067)
        
        # Testing the type of a for loop iterable (line 195)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 8), tuple_13063)
        # Getting the type of the for loop variable (line 195)
        for_loop_var_13068 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 8), tuple_13063)
        # Assigning a type to the variable 'tp' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'tp', for_loop_var_13068)
        # SSA begins for a for statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 196):
        
        # Assigning a Subscript to a Name (line 196):
        
        # Obtaining the type of the subscript
        int_13069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 53), 'int')
        
        # Call to kmeans(...): (line 196)
        # Processing the call arguments (line 196)
        
        # Call to tp(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'X' (line 196)
        X_13072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 30), 'X', False)
        # Processing the call keyword arguments (line 196)
        kwargs_13073 = {}
        # Getting the type of 'tp' (line 196)
        tp_13071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 27), 'tp', False)
        # Calling tp(args, kwargs) (line 196)
        tp_call_result_13074 = invoke(stypy.reporting.localization.Localization(__file__, 196, 27), tp_13071, *[X_13072], **kwargs_13073)
        
        
        # Call to tp(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'initc' (line 196)
        initc_13076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 37), 'initc', False)
        # Processing the call keyword arguments (line 196)
        kwargs_13077 = {}
        # Getting the type of 'tp' (line 196)
        tp_13075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 34), 'tp', False)
        # Calling tp(args, kwargs) (line 196)
        tp_call_result_13078 = invoke(stypy.reporting.localization.Localization(__file__, 196, 34), tp_13075, *[initc_13076], **kwargs_13077)
        
        # Processing the call keyword arguments (line 196)
        int_13079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 50), 'int')
        keyword_13080 = int_13079
        kwargs_13081 = {'iter': keyword_13080}
        # Getting the type of 'kmeans' (line 196)
        kmeans_13070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'kmeans', False)
        # Calling kmeans(args, kwargs) (line 196)
        kmeans_call_result_13082 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), kmeans_13070, *[tp_call_result_13074, tp_call_result_13078], **kwargs_13081)
        
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___13083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 20), kmeans_call_result_13082, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_13084 = invoke(stypy.reporting.localization.Localization(__file__, 196, 20), getitem___13083, int_13069)
        
        # Assigning a type to the variable 'code1' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'code1', subscript_call_result_13084)
        
        # Call to assert_array_almost_equal(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'code1' (line 197)
        code1_13086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 38), 'code1', False)
        # Getting the type of 'CODET2' (line 197)
        CODET2_13087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 45), 'CODET2', False)
        # Processing the call keyword arguments (line 197)
        kwargs_13088 = {}
        # Getting the type of 'assert_array_almost_equal' (line 197)
        assert_array_almost_equal_13085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 197)
        assert_array_almost_equal_call_result_13089 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), assert_array_almost_equal_13085, *[code1_13086, CODET2_13087], **kwargs_13088)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_kmeans_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_13090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13090)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans_simple'
        return stypy_return_type_13090


    @norecursion
    def test_kmeans_lost_cluster(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans_lost_cluster'
        module_type_store = module_type_store.open_function_context('test_kmeans_lost_cluster', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans_lost_cluster')
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans_lost_cluster.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans_lost_cluster', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans_lost_cluster', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans_lost_cluster(...)' code ##################

        
        # Assigning a Name to a Name (line 201):
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'TESTDATA_2D' (line 201)
        TESTDATA_2D_13091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'data', TESTDATA_2D_13091)
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to array(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_13094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_13095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        float_13096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_13095, float_13096)
        # Adding element type (line 202)
        float_13097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), list_13095, float_13097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 25), list_13094, list_13095)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_13098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        # Adding element type (line 203)
        float_13099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_13098, float_13099)
        # Adding element type (line 203)
        float_13100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 25), list_13098, float_13100)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 25), list_13094, list_13098)
        # Adding element type (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_13101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        float_13102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), list_13101, float_13102)
        # Adding element type (line 204)
        float_13103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), list_13101, float_13103)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 25), list_13094, list_13101)
        
        # Processing the call keyword arguments (line 202)
        kwargs_13104 = {}
        # Getting the type of 'np' (line 202)
        np_13092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'np', False)
        # Obtaining the member 'array' of a type (line 202)
        array_13093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), np_13092, 'array')
        # Calling array(args, kwargs) (line 202)
        array_call_result_13105 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), array_13093, *[list_13094], **kwargs_13104)
        
        # Assigning a type to the variable 'initk' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'initk', array_call_result_13105)
        
        # Call to kmeans(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'data' (line 206)
        data_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'data', False)
        # Getting the type of 'initk' (line 206)
        initk_13108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 21), 'initk', False)
        # Processing the call keyword arguments (line 206)
        kwargs_13109 = {}
        # Getting the type of 'kmeans' (line 206)
        kmeans_13106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'kmeans', False)
        # Calling kmeans(args, kwargs) (line 206)
        kmeans_call_result_13110 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), kmeans_13106, *[data_13107, initk_13108], **kwargs_13109)
        
        
        # Call to suppress_warnings(...): (line 207)
        # Processing the call keyword arguments (line 207)
        kwargs_13112 = {}
        # Getting the type of 'suppress_warnings' (line 207)
        suppress_warnings_13111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 207)
        suppress_warnings_call_result_13113 = invoke(stypy.reporting.localization.Localization(__file__, 207, 13), suppress_warnings_13111, *[], **kwargs_13112)
        
        with_13114 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 207, 13), suppress_warnings_call_result_13113, 'with parameter', '__enter__', '__exit__')

        if with_13114:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 207)
            enter___13115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 13), suppress_warnings_call_result_13113, '__enter__')
            with_enter_13116 = invoke(stypy.reporting.localization.Localization(__file__, 207, 13), enter___13115)
            # Assigning a type to the variable 'sup' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 13), 'sup', with_enter_13116)
            
            # Call to filter(...): (line 208)
            # Processing the call arguments (line 208)
            # Getting the type of 'UserWarning' (line 208)
            UserWarning_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 23), 'UserWarning', False)
            str_13120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'str', 'One of the clusters is empty. Re-run kmeans with a different initialization')
            # Processing the call keyword arguments (line 208)
            kwargs_13121 = {}
            # Getting the type of 'sup' (line 208)
            sup_13117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 208)
            filter_13118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), sup_13117, 'filter')
            # Calling filter(args, kwargs) (line 208)
            filter_call_result_13122 = invoke(stypy.reporting.localization.Localization(__file__, 208, 12), filter_13118, *[UserWarning_13119, str_13120], **kwargs_13121)
            
            
            # Call to kmeans2(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'data' (line 211)
            data_13124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'data', False)
            # Getting the type of 'initk' (line 211)
            initk_13125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 26), 'initk', False)
            # Processing the call keyword arguments (line 211)
            str_13126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 41), 'str', 'warn')
            keyword_13127 = str_13126
            kwargs_13128 = {'missing': keyword_13127}
            # Getting the type of 'kmeans2' (line 211)
            kmeans2_13123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'kmeans2', False)
            # Calling kmeans2(args, kwargs) (line 211)
            kmeans2_call_result_13129 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), kmeans2_13123, *[data_13124, initk_13125], **kwargs_13128)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 207)
            exit___13130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 13), suppress_warnings_call_result_13113, '__exit__')
            with_exit_13131 = invoke(stypy.reporting.localization.Localization(__file__, 207, 13), exit___13130, None, None, None)

        
        # Call to assert_raises(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'ClusterError' (line 213)
        ClusterError_13133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'ClusterError', False)
        # Getting the type of 'kmeans2' (line 213)
        kmeans2_13134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'kmeans2', False)
        # Getting the type of 'data' (line 213)
        data_13135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 45), 'data', False)
        # Getting the type of 'initk' (line 213)
        initk_13136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 51), 'initk', False)
        # Processing the call keyword arguments (line 213)
        str_13137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 66), 'str', 'raise')
        keyword_13138 = str_13137
        kwargs_13139 = {'missing': keyword_13138}
        # Getting the type of 'assert_raises' (line 213)
        assert_raises_13132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 213)
        assert_raises_call_result_13140 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), assert_raises_13132, *[ClusterError_13133, kmeans2_13134, data_13135, initk_13136], **kwargs_13139)
        
        
        # ################# End of 'test_kmeans_lost_cluster(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans_lost_cluster' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_13141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13141)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans_lost_cluster'
        return stypy_return_type_13141


    @norecursion
    def test_kmeans2_simple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_simple'
        module_type_store = module_type_store.open_function_context('test_kmeans2_simple', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_simple')
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_simple.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_simple', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_simple', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_simple(...)' code ##################

        
        # Call to seed(...): (line 216)
        # Processing the call arguments (line 216)
        int_13145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'int')
        # Processing the call keyword arguments (line 216)
        kwargs_13146 = {}
        # Getting the type of 'np' (line 216)
        np_13142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 216)
        random_13143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), np_13142, 'random')
        # Obtaining the member 'seed' of a type (line 216)
        seed_13144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), random_13143, 'seed')
        # Calling seed(args, kwargs) (line 216)
        seed_call_result_13147 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), seed_13144, *[int_13145], **kwargs_13146)
        
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to concatenate(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_13150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_13151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining the type of the subscript
        int_13152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'int')
        # Getting the type of 'X' (line 217)
        X_13153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___13154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 34), X_13153, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_13155 = invoke(stypy.reporting.localization.Localization(__file__, 217, 34), getitem___13154, int_13152)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 33), list_13151, subscript_call_result_13155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 32), list_13150, list_13151)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_13156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining the type of the subscript
        int_13157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'int')
        # Getting the type of 'X' (line 217)
        X_13158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 42), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___13159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 42), X_13158, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_13160 = invoke(stypy.reporting.localization.Localization(__file__, 217, 42), getitem___13159, int_13157)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 41), list_13156, subscript_call_result_13160)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 32), list_13150, list_13156)
        # Adding element type (line 217)
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_13161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        
        # Obtaining the type of the subscript
        int_13162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 52), 'int')
        # Getting the type of 'X' (line 217)
        X_13163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 50), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___13164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 50), X_13163, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_13165 = invoke(stypy.reporting.localization.Localization(__file__, 217, 50), getitem___13164, int_13162)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 49), list_13161, subscript_call_result_13165)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 32), list_13150, list_13161)
        
        # Processing the call keyword arguments (line 217)
        kwargs_13166 = {}
        # Getting the type of 'np' (line 217)
        np_13148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 217)
        concatenate_13149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), np_13148, 'concatenate')
        # Calling concatenate(args, kwargs) (line 217)
        concatenate_call_result_13167 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), concatenate_13149, *[list_13150], **kwargs_13166)
        
        # Assigning a type to the variable 'initc' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'initc', concatenate_call_result_13167)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_13168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        # Getting the type of 'np' (line 218)
        np_13169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'np')
        # Obtaining the member 'array' of a type (line 218)
        array_13170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 18), np_13169, 'array')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 18), tuple_13168, array_13170)
        # Adding element type (line 218)
        # Getting the type of 'np' (line 218)
        np_13171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'np')
        # Obtaining the member 'matrix' of a type (line 218)
        matrix_13172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 28), np_13171, 'matrix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 18), tuple_13168, matrix_13172)
        
        # Testing the type of a for loop iterable (line 218)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 8), tuple_13168)
        # Getting the type of the for loop variable (line 218)
        for_loop_var_13173 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 8), tuple_13168)
        # Assigning a type to the variable 'tp' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tp', for_loop_var_13173)
        # SSA begins for a for statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 219):
        
        # Assigning a Subscript to a Name (line 219):
        
        # Obtaining the type of the subscript
        int_13174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 54), 'int')
        
        # Call to kmeans2(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Call to tp(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'X' (line 219)
        X_13177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'X', False)
        # Processing the call keyword arguments (line 219)
        kwargs_13178 = {}
        # Getting the type of 'tp' (line 219)
        tp_13176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'tp', False)
        # Calling tp(args, kwargs) (line 219)
        tp_call_result_13179 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), tp_13176, *[X_13177], **kwargs_13178)
        
        
        # Call to tp(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'initc' (line 219)
        initc_13181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 38), 'initc', False)
        # Processing the call keyword arguments (line 219)
        kwargs_13182 = {}
        # Getting the type of 'tp' (line 219)
        tp_13180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 35), 'tp', False)
        # Calling tp(args, kwargs) (line 219)
        tp_call_result_13183 = invoke(stypy.reporting.localization.Localization(__file__, 219, 35), tp_13180, *[initc_13181], **kwargs_13182)
        
        # Processing the call keyword arguments (line 219)
        int_13184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 51), 'int')
        keyword_13185 = int_13184
        kwargs_13186 = {'iter': keyword_13185}
        # Getting the type of 'kmeans2' (line 219)
        kmeans2_13175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 219)
        kmeans2_call_result_13187 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), kmeans2_13175, *[tp_call_result_13179, tp_call_result_13183], **kwargs_13186)
        
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___13188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 20), kmeans2_call_result_13187, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_13189 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), getitem___13188, int_13174)
        
        # Assigning a type to the variable 'code1' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'code1', subscript_call_result_13189)
        
        # Assigning a Subscript to a Name (line 220):
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_13190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 54), 'int')
        
        # Call to kmeans2(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to tp(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'X' (line 220)
        X_13193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'X', False)
        # Processing the call keyword arguments (line 220)
        kwargs_13194 = {}
        # Getting the type of 'tp' (line 220)
        tp_13192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'tp', False)
        # Calling tp(args, kwargs) (line 220)
        tp_call_result_13195 = invoke(stypy.reporting.localization.Localization(__file__, 220, 28), tp_13192, *[X_13193], **kwargs_13194)
        
        
        # Call to tp(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'initc' (line 220)
        initc_13197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'initc', False)
        # Processing the call keyword arguments (line 220)
        kwargs_13198 = {}
        # Getting the type of 'tp' (line 220)
        tp_13196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'tp', False)
        # Calling tp(args, kwargs) (line 220)
        tp_call_result_13199 = invoke(stypy.reporting.localization.Localization(__file__, 220, 35), tp_13196, *[initc_13197], **kwargs_13198)
        
        # Processing the call keyword arguments (line 220)
        int_13200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 51), 'int')
        keyword_13201 = int_13200
        kwargs_13202 = {'iter': keyword_13201}
        # Getting the type of 'kmeans2' (line 220)
        kmeans2_13191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 220)
        kmeans2_call_result_13203 = invoke(stypy.reporting.localization.Localization(__file__, 220, 20), kmeans2_13191, *[tp_call_result_13195, tp_call_result_13199], **kwargs_13202)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___13204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 20), kmeans2_call_result_13203, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_13205 = invoke(stypy.reporting.localization.Localization(__file__, 220, 20), getitem___13204, int_13190)
        
        # Assigning a type to the variable 'code2' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'code2', subscript_call_result_13205)
        
        # Call to assert_array_almost_equal(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'code1' (line 222)
        code1_13207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 38), 'code1', False)
        # Getting the type of 'CODET1' (line 222)
        CODET1_13208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 45), 'CODET1', False)
        # Processing the call keyword arguments (line 222)
        kwargs_13209 = {}
        # Getting the type of 'assert_array_almost_equal' (line 222)
        assert_array_almost_equal_13206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 222)
        assert_array_almost_equal_call_result_13210 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), assert_array_almost_equal_13206, *[code1_13207, CODET1_13208], **kwargs_13209)
        
        
        # Call to assert_array_almost_equal(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'code2' (line 223)
        code2_13212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 38), 'code2', False)
        # Getting the type of 'CODET2' (line 223)
        CODET2_13213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 45), 'CODET2', False)
        # Processing the call keyword arguments (line 223)
        kwargs_13214 = {}
        # Getting the type of 'assert_array_almost_equal' (line 223)
        assert_array_almost_equal_13211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 223)
        assert_array_almost_equal_call_result_13215 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), assert_array_almost_equal_13211, *[code2_13212, CODET2_13213], **kwargs_13214)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_kmeans2_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_13216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_simple'
        return stypy_return_type_13216


    @norecursion
    def test_kmeans2_rank1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_rank1'
        module_type_store = module_type_store.open_function_context('test_kmeans2_rank1', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_rank1')
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_rank1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_rank1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_rank1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_rank1(...)' code ##################

        
        # Assigning a Name to a Name (line 226):
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'TESTDATA_2D' (line 226)
        TESTDATA_2D_13217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'data', TESTDATA_2D_13217)
        
        # Assigning a Subscript to a Name (line 227):
        
        # Assigning a Subscript to a Name (line 227):
        
        # Obtaining the type of the subscript
        slice_13218 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 227, 16), None, None, None)
        int_13219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 24), 'int')
        # Getting the type of 'data' (line 227)
        data_13220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___13221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), data_13220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_13222 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), getitem___13221, (slice_13218, int_13219))
        
        # Assigning a type to the variable 'data1' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'data1', subscript_call_result_13222)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        int_13223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'int')
        slice_13224 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 229, 16), None, int_13223, None)
        # Getting the type of 'data1' (line 229)
        data1_13225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'data1')
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___13226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), data1_13225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_13227 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), getitem___13226, slice_13224)
        
        # Assigning a type to the variable 'initc' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'initc', subscript_call_result_13227)
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to copy(...): (line 230)
        # Processing the call keyword arguments (line 230)
        kwargs_13230 = {}
        # Getting the type of 'initc' (line 230)
        initc_13228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'initc', False)
        # Obtaining the member 'copy' of a type (line 230)
        copy_13229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 15), initc_13228, 'copy')
        # Calling copy(args, kwargs) (line 230)
        copy_call_result_13231 = invoke(stypy.reporting.localization.Localization(__file__, 230, 15), copy_13229, *[], **kwargs_13230)
        
        # Assigning a type to the variable 'code' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'code', copy_call_result_13231)
        
        # Obtaining the type of the subscript
        int_13232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 37), 'int')
        
        # Call to kmeans2(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'data1' (line 231)
        data1_13234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'data1', False)
        # Getting the type of 'code' (line 231)
        code_13235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'code', False)
        # Processing the call keyword arguments (line 231)
        int_13236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'int')
        keyword_13237 = int_13236
        kwargs_13238 = {'iter': keyword_13237}
        # Getting the type of 'kmeans2' (line 231)
        kmeans2_13233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 231)
        kmeans2_call_result_13239 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), kmeans2_13233, *[data1_13234, code_13235], **kwargs_13238)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___13240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), kmeans2_call_result_13239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_13241 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___13240, int_13232)
        
        
        # Obtaining the type of the subscript
        int_13242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 37), 'int')
        
        # Call to kmeans2(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'data1' (line 232)
        data1_13244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'data1', False)
        # Getting the type of 'code' (line 232)
        code_13245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'code', False)
        # Processing the call keyword arguments (line 232)
        int_13246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 34), 'int')
        keyword_13247 = int_13246
        kwargs_13248 = {'iter': keyword_13247}
        # Getting the type of 'kmeans2' (line 232)
        kmeans2_13243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 232)
        kmeans2_call_result_13249 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), kmeans2_13243, *[data1_13244, code_13245], **kwargs_13248)
        
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___13250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), kmeans2_call_result_13249, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_13251 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), getitem___13250, int_13242)
        
        
        # ################# End of 'test_kmeans2_rank1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_rank1' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_13252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_rank1'
        return stypy_return_type_13252


    @norecursion
    def test_kmeans2_rank1_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_rank1_2'
        module_type_store = module_type_store.open_function_context('test_kmeans2_rank1_2', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_rank1_2')
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_rank1_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_rank1_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_rank1_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_rank1_2(...)' code ##################

        
        # Assigning a Name to a Name (line 235):
        
        # Assigning a Name to a Name (line 235):
        # Getting the type of 'TESTDATA_2D' (line 235)
        TESTDATA_2D_13253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'data', TESTDATA_2D_13253)
        
        # Assigning a Subscript to a Name (line 236):
        
        # Assigning a Subscript to a Name (line 236):
        
        # Obtaining the type of the subscript
        slice_13254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 236, 16), None, None, None)
        int_13255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 24), 'int')
        # Getting the type of 'data' (line 236)
        data_13256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___13257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), data_13256, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_13258 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), getitem___13257, (slice_13254, int_13255))
        
        # Assigning a type to the variable 'data1' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'data1', subscript_call_result_13258)
        
        # Call to kmeans2(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'data1' (line 237)
        data1_13260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'data1', False)
        int_13261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 23), 'int')
        # Processing the call keyword arguments (line 237)
        int_13262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 31), 'int')
        keyword_13263 = int_13262
        kwargs_13264 = {'iter': keyword_13263}
        # Getting the type of 'kmeans2' (line 237)
        kmeans2_13259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 237)
        kmeans2_call_result_13265 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), kmeans2_13259, *[data1_13260, int_13261], **kwargs_13264)
        
        
        # ################# End of 'test_kmeans2_rank1_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_rank1_2' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_13266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_rank1_2'
        return stypy_return_type_13266


    @norecursion
    def test_kmeans2_high_dim(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_high_dim'
        module_type_store = module_type_store.open_function_context('test_kmeans2_high_dim', 239, 4, False)
        # Assigning a type to the variable 'self' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_high_dim')
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_high_dim.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_high_dim', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_high_dim', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_high_dim(...)' code ##################

        
        # Assigning a Name to a Name (line 242):
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'TESTDATA_2D' (line 242)
        TESTDATA_2D_13267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'data', TESTDATA_2D_13267)
        
        # Assigning a Subscript to a Name (line 243):
        
        # Assigning a Subscript to a Name (line 243):
        
        # Obtaining the type of the subscript
        int_13268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 39), 'int')
        slice_13269 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 243, 15), None, int_13268, None)
        
        # Call to reshape(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_13272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        int_13273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 29), tuple_13272, int_13273)
        # Adding element type (line 243)
        int_13274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 29), tuple_13272, int_13274)
        
        # Processing the call keyword arguments (line 243)
        kwargs_13275 = {}
        # Getting the type of 'data' (line 243)
        data_13270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'data', False)
        # Obtaining the member 'reshape' of a type (line 243)
        reshape_13271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), data_13270, 'reshape')
        # Calling reshape(args, kwargs) (line 243)
        reshape_call_result_13276 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), reshape_13271, *[tuple_13272], **kwargs_13275)
        
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___13277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), reshape_call_result_13276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_13278 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), getitem___13277, slice_13269)
        
        # Assigning a type to the variable 'data' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'data', subscript_call_result_13278)
        
        # Call to kmeans2(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'data' (line 244)
        data_13280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'data', False)
        int_13281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 22), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_13282 = {}
        # Getting the type of 'kmeans2' (line 244)
        kmeans2_13279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 244)
        kmeans2_call_result_13283 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), kmeans2_13279, *[data_13280, int_13281], **kwargs_13282)
        
        
        # ################# End of 'test_kmeans2_high_dim(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_high_dim' in the type store
        # Getting the type of 'stypy_return_type' (line 239)
        stypy_return_type_13284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_high_dim'
        return stypy_return_type_13284


    @norecursion
    def test_kmeans2_init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_init'
        module_type_store = module_type_store.open_function_context('test_kmeans2_init', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_init')
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_init.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_init', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_init', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_init(...)' code ##################

        
        # Call to seed(...): (line 247)
        # Processing the call arguments (line 247)
        int_13288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'int')
        # Processing the call keyword arguments (line 247)
        kwargs_13289 = {}
        # Getting the type of 'np' (line 247)
        np_13285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 247)
        random_13286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), np_13285, 'random')
        # Obtaining the member 'seed' of a type (line 247)
        seed_13287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), random_13286, 'seed')
        # Calling seed(args, kwargs) (line 247)
        seed_call_result_13290 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), seed_13287, *[int_13288], **kwargs_13289)
        
        
        # Assigning a Name to a Name (line 248):
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'TESTDATA_2D' (line 248)
        TESTDATA_2D_13291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'data', TESTDATA_2D_13291)
        
        # Call to kmeans2(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'data' (line 250)
        data_13293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'data', False)
        int_13294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 22), 'int')
        # Processing the call keyword arguments (line 250)
        str_13295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 31), 'str', 'points')
        keyword_13296 = str_13295
        kwargs_13297 = {'minit': keyword_13296}
        # Getting the type of 'kmeans2' (line 250)
        kmeans2_13292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 250)
        kmeans2_call_result_13298 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), kmeans2_13292, *[data_13293, int_13294], **kwargs_13297)
        
        
        # Call to kmeans2(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Obtaining the type of the subscript
        slice_13300 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 16), None, None, None)
        int_13301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'int')
        slice_13302 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 251, 16), None, int_13301, None)
        # Getting the type of 'data' (line 251)
        data_13303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___13304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 16), data_13303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_13305 = invoke(stypy.reporting.localization.Localization(__file__, 251, 16), getitem___13304, (slice_13300, slice_13302))
        
        int_13306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'int')
        # Processing the call keyword arguments (line 251)
        str_13307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 38), 'str', 'points')
        keyword_13308 = str_13307
        kwargs_13309 = {'minit': keyword_13308}
        # Getting the type of 'kmeans2' (line 251)
        kmeans2_13299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 251)
        kmeans2_call_result_13310 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), kmeans2_13299, *[subscript_call_result_13305, int_13306], **kwargs_13309)
        
        
        # Call to kmeans2(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'data' (line 253)
        data_13312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'data', False)
        int_13313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'int')
        # Processing the call keyword arguments (line 253)
        str_13314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 31), 'str', 'random')
        keyword_13315 = str_13314
        kwargs_13316 = {'minit': keyword_13315}
        # Getting the type of 'kmeans2' (line 253)
        kmeans2_13311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 253)
        kmeans2_call_result_13317 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), kmeans2_13311, *[data_13312, int_13313], **kwargs_13316)
        
        
        # Call to kmeans2(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining the type of the subscript
        slice_13319 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 254, 16), None, None, None)
        int_13320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 25), 'int')
        slice_13321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 254, 16), None, int_13320, None)
        # Getting the type of 'data' (line 254)
        data_13322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___13323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 16), data_13322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_13324 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), getitem___13323, (slice_13319, slice_13321))
        
        int_13325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'int')
        # Processing the call keyword arguments (line 254)
        str_13326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 38), 'str', 'random')
        keyword_13327 = str_13326
        kwargs_13328 = {'minit': keyword_13327}
        # Getting the type of 'kmeans2' (line 254)
        kmeans2_13318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'kmeans2', False)
        # Calling kmeans2(args, kwargs) (line 254)
        kmeans2_call_result_13329 = invoke(stypy.reporting.localization.Localization(__file__, 254, 8), kmeans2_13318, *[subscript_call_result_13324, int_13325], **kwargs_13328)
        
        
        # ################# End of 'test_kmeans2_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_init' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_13330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_init'
        return stypy_return_type_13330


    @norecursion
    def test_krandinit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_krandinit'
        module_type_store = module_type_store.open_function_context('test_krandinit', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_krandinit')
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_krandinit.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_krandinit', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_krandinit', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_krandinit(...)' code ##################

        
        # Assigning a Name to a Name (line 258):
        
        # Assigning a Name to a Name (line 258):
        # Getting the type of 'TESTDATA_2D' (line 258)
        TESTDATA_2D_13331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'TESTDATA_2D')
        # Assigning a type to the variable 'data' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'data', TESTDATA_2D_13331)
        
        # Assigning a List to a Name (line 259):
        
        # Assigning a List to a Name (line 259):
        
        # Obtaining an instance of the builtin type 'list' (line 259)
        list_13332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 259)
        # Adding element type (line 259)
        
        # Call to reshape(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_13335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        int_13336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 31), tuple_13335, int_13336)
        # Adding element type (line 259)
        int_13337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 31), tuple_13335, int_13337)
        
        # Processing the call keyword arguments (line 259)
        kwargs_13338 = {}
        # Getting the type of 'data' (line 259)
        data_13333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'data', False)
        # Obtaining the member 'reshape' of a type (line 259)
        reshape_13334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 17), data_13333, 'reshape')
        # Calling reshape(args, kwargs) (line 259)
        reshape_call_result_13339 = invoke(stypy.reporting.localization.Localization(__file__, 259, 17), reshape_13334, *[tuple_13335], **kwargs_13338)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 16), list_13332, reshape_call_result_13339)
        # Adding element type (line 259)
        
        # Obtaining the type of the subscript
        int_13340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 65), 'int')
        slice_13341 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 41), None, int_13340, None)
        
        # Call to reshape(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_13344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        int_13345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 55), tuple_13344, int_13345)
        # Adding element type (line 259)
        int_13346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 55), tuple_13344, int_13346)
        
        # Processing the call keyword arguments (line 259)
        kwargs_13347 = {}
        # Getting the type of 'data' (line 259)
        data_13342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 41), 'data', False)
        # Obtaining the member 'reshape' of a type (line 259)
        reshape_13343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 41), data_13342, 'reshape')
        # Calling reshape(args, kwargs) (line 259)
        reshape_call_result_13348 = invoke(stypy.reporting.localization.Localization(__file__, 259, 41), reshape_13343, *[tuple_13344], **kwargs_13347)
        
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___13349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 41), reshape_call_result_13348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_13350 = invoke(stypy.reporting.localization.Localization(__file__, 259, 41), getitem___13349, slice_13341)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 16), list_13332, subscript_call_result_13350)
        
        # Assigning a type to the variable 'datas' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'datas', list_13332)
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to int(...): (line 260)
        # Processing the call arguments (line 260)
        float_13352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 16), 'float')
        # Processing the call keyword arguments (line 260)
        kwargs_13353 = {}
        # Getting the type of 'int' (line 260)
        int_13351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'int', False)
        # Calling int(args, kwargs) (line 260)
        int_call_result_13354 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), int_13351, *[float_13352], **kwargs_13353)
        
        # Assigning a type to the variable 'k' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'k', int_call_result_13354)
        
        # Getting the type of 'datas' (line 261)
        datas_13355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'datas')
        # Testing the type of a for loop iterable (line 261)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 261, 8), datas_13355)
        # Getting the type of the for loop variable (line 261)
        for_loop_var_13356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 261, 8), datas_13355)
        # Assigning a type to the variable 'data' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'data', for_loop_var_13356)
        # SSA begins for a for statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to seed(...): (line 262)
        # Processing the call arguments (line 262)
        int_13360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 27), 'int')
        # Processing the call keyword arguments (line 262)
        kwargs_13361 = {}
        # Getting the type of 'np' (line 262)
        np_13357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 262)
        random_13358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), np_13357, 'random')
        # Obtaining the member 'seed' of a type (line 262)
        seed_13359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), random_13358, 'seed')
        # Calling seed(args, kwargs) (line 262)
        seed_call_result_13362 = invoke(stypy.reporting.localization.Localization(__file__, 262, 12), seed_13359, *[int_13360], **kwargs_13361)
        
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to _krandinit(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'data' (line 263)
        data_13364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'data', False)
        # Getting the type of 'k' (line 263)
        k_13365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 36), 'k', False)
        # Processing the call keyword arguments (line 263)
        kwargs_13366 = {}
        # Getting the type of '_krandinit' (line 263)
        _krandinit_13363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), '_krandinit', False)
        # Calling _krandinit(args, kwargs) (line 263)
        _krandinit_call_result_13367 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), _krandinit_13363, *[data_13364, k_13365], **kwargs_13366)
        
        # Assigning a type to the variable 'init' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'init', _krandinit_call_result_13367)
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to cov(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'data' (line 264)
        data_13370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'data', False)
        # Processing the call keyword arguments (line 264)
        int_13371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 43), 'int')
        keyword_13372 = int_13371
        kwargs_13373 = {'rowvar': keyword_13372}
        # Getting the type of 'np' (line 264)
        np_13368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'np', False)
        # Obtaining the member 'cov' of a type (line 264)
        cov_13369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 23), np_13368, 'cov')
        # Calling cov(args, kwargs) (line 264)
        cov_call_result_13374 = invoke(stypy.reporting.localization.Localization(__file__, 264, 23), cov_13369, *[data_13370], **kwargs_13373)
        
        # Assigning a type to the variable 'orig_cov' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'orig_cov', cov_call_result_13374)
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to cov(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'init' (line 265)
        init_13377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'init', False)
        # Processing the call keyword arguments (line 265)
        int_13378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 43), 'int')
        keyword_13379 = int_13378
        kwargs_13380 = {'rowvar': keyword_13379}
        # Getting the type of 'np' (line 265)
        np_13375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'np', False)
        # Obtaining the member 'cov' of a type (line 265)
        cov_13376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 23), np_13375, 'cov')
        # Calling cov(args, kwargs) (line 265)
        cov_call_result_13381 = invoke(stypy.reporting.localization.Localization(__file__, 265, 23), cov_13376, *[init_13377], **kwargs_13380)
        
        # Assigning a type to the variable 'init_cov' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'init_cov', cov_call_result_13381)
        
        # Call to assert_allclose(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'orig_cov' (line 266)
        orig_cov_13383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'orig_cov', False)
        # Getting the type of 'init_cov' (line 266)
        init_cov_13384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'init_cov', False)
        # Processing the call keyword arguments (line 266)
        float_13385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 53), 'float')
        keyword_13386 = float_13385
        kwargs_13387 = {'atol': keyword_13386}
        # Getting the type of 'assert_allclose' (line 266)
        assert_allclose_13382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 266)
        assert_allclose_call_result_13388 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), assert_allclose_13382, *[orig_cov_13383, init_cov_13384], **kwargs_13387)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_krandinit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_krandinit' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_13389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_krandinit'
        return stypy_return_type_13389


    @norecursion
    def test_kmeans2_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans2_empty'
        module_type_store = module_type_store.open_function_context('test_kmeans2_empty', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans2_empty')
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans2_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans2_empty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans2_empty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans2_empty(...)' code ##################

        
        # Call to assert_raises(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'ValueError' (line 270)
        ValueError_13391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'ValueError', False)
        # Getting the type of 'kmeans2' (line 270)
        kmeans2_13392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 34), 'kmeans2', False)
        
        # Obtaining an instance of the builtin type 'list' (line 270)
        list_13393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 270)
        
        int_13394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 47), 'int')
        # Processing the call keyword arguments (line 270)
        kwargs_13395 = {}
        # Getting the type of 'assert_raises' (line 270)
        assert_raises_13390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 270)
        assert_raises_call_result_13396 = invoke(stypy.reporting.localization.Localization(__file__, 270, 8), assert_raises_13390, *[ValueError_13391, kmeans2_13392, list_13393, int_13394], **kwargs_13395)
        
        
        # ################# End of 'test_kmeans2_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans2_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_13397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans2_empty'
        return stypy_return_type_13397


    @norecursion
    def test_kmeans_0k(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans_0k'
        module_type_store = module_type_store.open_function_context('test_kmeans_0k', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans_0k')
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans_0k.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans_0k', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans_0k', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans_0k(...)' code ##################

        
        # Call to assert_raises(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'ValueError' (line 274)
        ValueError_13399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'ValueError', False)
        # Getting the type of 'kmeans' (line 274)
        kmeans_13400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 34), 'kmeans', False)
        # Getting the type of 'X' (line 274)
        X_13401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 42), 'X', False)
        int_13402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'int')
        # Processing the call keyword arguments (line 274)
        kwargs_13403 = {}
        # Getting the type of 'assert_raises' (line 274)
        assert_raises_13398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 274)
        assert_raises_call_result_13404 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), assert_raises_13398, *[ValueError_13399, kmeans_13400, X_13401, int_13402], **kwargs_13403)
        
        
        # Call to assert_raises(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'ValueError' (line 275)
        ValueError_13406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'ValueError', False)
        # Getting the type of 'kmeans2' (line 275)
        kmeans2_13407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 34), 'kmeans2', False)
        # Getting the type of 'X' (line 275)
        X_13408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 43), 'X', False)
        int_13409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 46), 'int')
        # Processing the call keyword arguments (line 275)
        kwargs_13410 = {}
        # Getting the type of 'assert_raises' (line 275)
        assert_raises_13405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 275)
        assert_raises_call_result_13411 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), assert_raises_13405, *[ValueError_13406, kmeans2_13407, X_13408, int_13409], **kwargs_13410)
        
        
        # Call to assert_raises(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'ValueError' (line 276)
        ValueError_13413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'ValueError', False)
        # Getting the type of 'kmeans2' (line 276)
        kmeans2_13414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'kmeans2', False)
        # Getting the type of 'X' (line 276)
        X_13415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 43), 'X', False)
        
        # Call to array(...): (line 276)
        # Processing the call arguments (line 276)
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_13418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        
        # Processing the call keyword arguments (line 276)
        kwargs_13419 = {}
        # Getting the type of 'np' (line 276)
        np_13416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 276)
        array_13417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 46), np_13416, 'array')
        # Calling array(args, kwargs) (line 276)
        array_call_result_13420 = invoke(stypy.reporting.localization.Localization(__file__, 276, 46), array_13417, *[list_13418], **kwargs_13419)
        
        # Processing the call keyword arguments (line 276)
        kwargs_13421 = {}
        # Getting the type of 'assert_raises' (line 276)
        assert_raises_13412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 276)
        assert_raises_call_result_13422 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), assert_raises_13412, *[ValueError_13413, kmeans2_13414, X_13415, array_call_result_13420], **kwargs_13421)
        
        
        # ################# End of 'test_kmeans_0k(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans_0k' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_13423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans_0k'
        return stypy_return_type_13423


    @norecursion
    def test_kmeans_large_thres(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_kmeans_large_thres'
        module_type_store = module_type_store.open_function_context('test_kmeans_large_thres', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_localization', localization)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_function_name', 'TestKMean.test_kmeans_large_thres')
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_param_names_list', [])
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestKMean.test_kmeans_large_thres.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.test_kmeans_large_thres', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_kmeans_large_thres', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_kmeans_large_thres(...)' code ##################

        
        # Assigning a Call to a Name (line 280):
        
        # Assigning a Call to a Name (line 280):
        
        # Call to array(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Obtaining an instance of the builtin type 'list' (line 280)
        list_13426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 280)
        # Adding element type (line 280)
        int_13427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 21), list_13426, int_13427)
        # Adding element type (line 280)
        int_13428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 21), list_13426, int_13428)
        # Adding element type (line 280)
        int_13429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 21), list_13426, int_13429)
        # Adding element type (line 280)
        int_13430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 21), list_13426, int_13430)
        # Adding element type (line 280)
        int_13431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 280, 21), list_13426, int_13431)
        
        # Processing the call keyword arguments (line 280)
        # Getting the type of 'float' (line 280)
        float_13432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 41), 'float', False)
        keyword_13433 = float_13432
        kwargs_13434 = {'dtype': keyword_13433}
        # Getting the type of 'np' (line 280)
        np_13424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 280)
        array_13425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), np_13424, 'array')
        # Calling array(args, kwargs) (line 280)
        array_call_result_13435 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), array_13425, *[list_13426], **kwargs_13434)
        
        # Assigning a type to the variable 'x' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'x', array_call_result_13435)
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to kmeans(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'x' (line 281)
        x_13437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'x', False)
        int_13438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 24), 'int')
        # Processing the call keyword arguments (line 281)
        float_13439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 34), 'float')
        keyword_13440 = float_13439
        kwargs_13441 = {'thresh': keyword_13440}
        # Getting the type of 'kmeans' (line 281)
        kmeans_13436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'kmeans', False)
        # Calling kmeans(args, kwargs) (line 281)
        kmeans_call_result_13442 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), kmeans_13436, *[x_13437, int_13438], **kwargs_13441)
        
        # Assigning a type to the variable 'res' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'res', kmeans_call_result_13442)
        
        # Call to assert_allclose(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining the type of the subscript
        int_13444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 28), 'int')
        # Getting the type of 'res' (line 282)
        res_13445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___13446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), res_13445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_13447 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), getitem___13446, int_13444)
        
        
        # Call to array(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'list' (line 282)
        list_13450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 282)
        # Adding element type (line 282)
        float_13451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 41), list_13450, float_13451)
        
        # Processing the call keyword arguments (line 282)
        kwargs_13452 = {}
        # Getting the type of 'np' (line 282)
        np_13448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 282)
        array_13449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 32), np_13448, 'array')
        # Calling array(args, kwargs) (line 282)
        array_call_result_13453 = invoke(stypy.reporting.localization.Localization(__file__, 282, 32), array_13449, *[list_13450], **kwargs_13452)
        
        # Processing the call keyword arguments (line 282)
        kwargs_13454 = {}
        # Getting the type of 'assert_allclose' (line 282)
        assert_allclose_13443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 282)
        assert_allclose_call_result_13455 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assert_allclose_13443, *[subscript_call_result_13447, array_call_result_13453], **kwargs_13454)
        
        
        # Call to assert_allclose(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Obtaining the type of the subscript
        int_13457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'int')
        # Getting the type of 'res' (line 283)
        res_13458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'res', False)
        # Obtaining the member '__getitem__' of a type (line 283)
        getitem___13459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 24), res_13458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 283)
        subscript_call_result_13460 = invoke(stypy.reporting.localization.Localization(__file__, 283, 24), getitem___13459, int_13457)
        
        float_13461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 32), 'float')
        # Processing the call keyword arguments (line 283)
        kwargs_13462 = {}
        # Getting the type of 'assert_allclose' (line 283)
        assert_allclose_13456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 283)
        assert_allclose_call_result_13463 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), assert_allclose_13456, *[subscript_call_result_13460, float_13461], **kwargs_13462)
        
        
        # ################# End of 'test_kmeans_large_thres(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_kmeans_large_thres' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_13464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_kmeans_large_thres'
        return stypy_return_type_13464


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestKMean.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestKMean' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'TestKMean', TestKMean)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
